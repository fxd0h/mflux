import functools
import logging
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class Fibo2TextImages:
    # Images of the text a caption asks for, drawn the way Fibo 2.0's training drew them (diffusers' text_images.py):
    # each text on its own, in one line, black on white, in Pillow's built-in font at size 20. A character the
    # built-in font lacks is drawn with the first fallback font that has it: training used Aileron, the full version
    # of the built-in font, then DejaVu Sans. Both ship in the model repo's fonts/ folder.
    FONT_SIZE = 20
    MAX_LINE_CHARS = 256
    FONT_FILES = ("Aileron-Regular.otf", "DejaVuSans.ttf")
    _PADDING = 2
    _LINE_SPACING = 4

    @staticmethod
    def draw(text: str, fonts_dir: Path | None) -> Image.Image:
        font = Fibo2TextImages._default_font()
        line = " ".join(text.split())
        lines = [line[i : i + Fibo2TextImages.MAX_LINE_CHARS] for i in range(0, len(line), Fibo2TextImages.MAX_LINE_CHARS)] or [""]  # fmt: off
        if all(ord(character) < 128 or Fibo2TextImages._has_glyph(font, character) for character in line):
            return Fibo2TextImages._draw_one_font(lines, font)
        fallbacks = Fibo2TextImages._fallback_fonts(None if fonts_dir is None else str(fonts_dir))
        if not fallbacks:
            logger.warning(
                "A caption text has characters Pillow's built-in font lacks, and the model's fonts/ folder was not "
                "found, so they are drawn as the built-in font draws them."
            )
        return Fibo2TextImages._draw_mixed(lines, [font, *fallbacks])

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _fallback_fonts(fonts_dir: str | None) -> tuple[ImageFont.FreeTypeFont, ...]:
        if fonts_dir is None:
            return ()
        paths = [Path(fonts_dir) / name for name in Fibo2TextImages.FONT_FILES]
        # The basic layout, as the built-in font, so the text does not set differently where libraqm is installed
        return tuple(
            ImageFont.truetype(str(path), Fibo2TextImages.FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)
            for path in paths
            if path.is_file()
        )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _default_font() -> ImageFont.FreeTypeFont:
        font = ImageFont.load_default(Fibo2TextImages.FONT_SIZE)
        if not isinstance(font, ImageFont.FreeTypeFont):
            raise ImportError("Drawing the images of a caption's text needs Pillow built with FreeType.")
        return font

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _has_glyph(font: ImageFont.FreeTypeFont, character: str) -> bool:
        # A font without a glyph for the character draws its "notdef" box, as it does for U+FFFE, which no font has
        try:
            notdef = font.getmask("￾")
            mask = font.getmask(character)
        except Exception:  # noqa: BLE001 - as in training, a font that cannot say is taken to have the glyph
            return True
        return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))

    @staticmethod
    def _font_runs(text: str, fonts: Sequence[ImageFont.FreeTypeFont]) -> list[tuple[str, ImageFont.FreeTypeFont]]:
        # The text split into runs drawn with the same font: the first font with a glyph for the character, or the
        # built-in font, the first, when no font has one
        runs: list[list] = []
        for character in text:
            font = fonts[0]
            if not (ord(character) < 128 or Fibo2TextImages._has_glyph(fonts[0], character)):
                font = next((f for f in fonts[1:] if Fibo2TextImages._has_glyph(f, character)), fonts[0])
            if runs and runs[-1][1] is font:
                runs[-1][0] += character
            else:
                runs.append([character, font])
        return [(run, font) for run, font in runs]

    @staticmethod
    def _draw_one_font(lines: list[str], font: ImageFont.FreeTypeFont) -> Image.Image:
        # Each line is placed by its bounding box, and is at least as tall as "Ag"
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
        min_box = draw.textbbox((0, 0), "Ag", font=font)
        min_line_height = max(1, min_box[3] - min_box[1])
        line_sizes = []
        max_width = 1
        total_height = 0
        for line in lines:
            box = draw.textbbox((0, 0), line or " ", font=font)
            line_height = max(min_line_height, box[3] - box[1])
            line_sizes.append((box, line_height))
            max_width = max(max_width, box[2] - box[0])
            total_height += line_height
        total_height += Fibo2TextImages._LINE_SPACING * (len(lines) - 1)
        padding = Fibo2TextImages._PADDING
        image = Image.new("RGB", (max_width + padding * 2, total_height + padding * 2), "white")
        draw = ImageDraw.Draw(image)
        y = padding
        for line, (box, line_height) in zip(lines, line_sizes):
            draw.text((padding - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
            y += line_height + Fibo2TextImages._LINE_SPACING
        return image

    @staticmethod
    def _draw_mixed(lines: list[str], fonts: Sequence[ImageFont.FreeTypeFont]) -> Image.Image:
        # Each run is drawn with its own font on the line's baseline, each line as tall as the built-in font's ascent
        # and descent
        ascent, descent = fonts[0].getmetrics()
        line_height = max(1, ascent + descent)
        runs_per_line = [Fibo2TextImages._font_runs(line or " ", fonts) for line in lines]
        width = max(max(1, round(sum(font.getlength(run) for run, font in runs))) for runs in runs_per_line)
        height = line_height * len(lines) + Fibo2TextImages._LINE_SPACING * (len(lines) - 1)
        padding = Fibo2TextImages._PADDING
        image = Image.new("RGB", (width + padding * 2, height + padding * 2), "white")
        draw = ImageDraw.Draw(image)
        y = padding
        for runs in runs_per_line:
            x = float(padding)
            for run, font in runs:
                draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
                x += font.getlength(run)
            y += line_height + Fibo2TextImages._LINE_SPACING
        return image
