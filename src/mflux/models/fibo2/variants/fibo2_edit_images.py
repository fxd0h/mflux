import functools
import math
from pathlib import Path

from PIL import Image

from mflux.utils.image_util import ImageUtil


class Fibo2EditImages:
    # The images to edit, prepared as Fibo 2.0 was trained on them (diffusers' BriaFibo2ImageProcessor)
    MAX_IMAGES = 5
    DEFAULT_AREA = 1024 * 1024
    SIZE_MULTIPLE = 16
    # Qwen3-VL sees each image at most this many pixels a side
    TEXT_ENCODER_IMAGE_SIZE = 768

    @staticmethod
    def load(image_paths: list[str | Path], mask_path: str | Path | None) -> list[Image.Image]:
        if not 1 <= len(image_paths) <= Fibo2EditImages.MAX_IMAGES:
            raise ValueError(f"Fibo 2.0 edits 1 to {Fibo2EditImages.MAX_IMAGES} images at once, got {len(image_paths)}.")  # fmt: off
        images = [ImageUtil.load_image(path) for path in image_paths]
        if mask_path is None:
            return images
        if len(images) != 1:
            raise ValueError("A mask needs exactly one image to edit.")
        mask = ImageUtil.load_image(mask_path, composite_alpha=False)
        if mask.size != images[0].size:
            raise ValueError(f"The mask is {mask.size[0]}x{mask.size[1]} but the image is {images[0].size[0]}x{images[0].size[1]}.")  # fmt: off
        return [Fibo2EditImages.grey_out(images[0], mask)]

    @staticmethod
    def crops(images: list[Image.Image], width: int, height: int) -> list[Image.Image]:
        # One image is cropped to the size of the result. Several images are each cropped to the size Fibo 2.0 was
        # trained at for their own aspect ratio.
        if len(images) == 1:
            return [Fibo2EditImages._resize_and_crop(images[0], width, height)]
        sizes = [Fibo2EditImages.trained_size(image.width / image.height) for image in images]
        return [Fibo2EditImages._resize_and_crop(image, w, h) for image, (w, h) in zip(images, sizes)]

    @staticmethod
    def text_encoder_copy(crop: Image.Image) -> Image.Image:
        copy = crop.copy()
        copy.thumbnail((Fibo2EditImages.TEXT_ENCODER_IMAGE_SIZE, Fibo2EditImages.TEXT_ENCODER_IMAGE_SIZE))
        return copy

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def trained_size(aspect_ratio: float, area: int = DEFAULT_AREA) -> tuple[int, int]:
        # Of the (width, height) sizes divisible by 16, at least 64 a side and within 10% of the area: the one that
        # crops the least from a picture with this aspect ratio, then the one closest to the area
        multiple = Fibo2EditImages.SIZE_MULTIPLE
        min_side = 4 * multiple
        sizes = [
            (width, height)
            for width in range(min_side, 2 * area // min_side, multiple)
            for height in range(min_side, 2 * area // width, multiple)
            if 0.9 * area <= width * height <= 1.1 * area
        ]

        def crop(size: tuple[int, int]) -> float:
            ratio = size[0] / size[1]
            return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)

        return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))

    @staticmethod
    def grey_out(image: Image.Image, mask: Image.Image) -> Image.Image:
        # The region the mask marks, white on black, in flat (128, 128, 128) grey through the mask made binary at 128,
        # at the image's own size, before the image is resized
        grey = Image.new("RGB", image.size, (128, 128, 128))
        binary_mask = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        return Image.composite(grey, image.convert("RGB"), binary_mask)

    @staticmethod
    def _resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
        # Scaled to cover width x height, keeping the aspect ratio, then cropped at the center
        image = image.convert("RGB")
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))
