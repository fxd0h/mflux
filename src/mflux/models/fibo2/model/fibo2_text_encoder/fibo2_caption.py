import json
import math

# The null caption Fibo 2.0 was trained with for classifier-free guidance: every key of the structured caption, every
# value empty. It is the negative prompt of the model card for edits.
NULL_CAPTION = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

# The negative prompt of the model card for text to image: soft, featureless, artifact-laden images
TEXT_TO_IMAGE_NEGATIVE = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, '
    'no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":'
    '"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture",'
    '"lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights",'
    '"lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, '
    'heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},'
    '"style_medium":"","context":"","artistic_style":""}'
)

# Where an image stands in a Qwen3-VL prompt
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


class Fibo2Caption:
    # The prompt as Fibo 2.0's training saw it, following diffusers' BriaFibo2Pipeline. A prompt is a structured JSON
    # caption, or plain text: a short description, or an edit's instruction alone.

    @staticmethod
    def parse(prompt: str) -> tuple[dict | str | None, str | None]:
        # A prompt's caption, in its trained form, and its edit instruction. A prompt that is not a JSON object comes
        # back as it is, with no instruction. An edit prompt's caption is its other keys or its "structured_caption"
        # key, a JSON caption or a short description.
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            return prompt, None
        if not isinstance(caption, dict):
            return prompt, None
        instruction = caption.pop("edit_instruction", None)
        if "structured_caption" in caption:
            caption = caption.pop("structured_caption")
            if not isinstance(caption, dict):
                return caption or None, instruction
        # Training captions hold no empty values and no aesthetic_score or preference_score (FIBO 1 had them)
        keys = set(caption)
        scores = ("aesthetic_score", "preference_score")
        caption = {
            key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value
            for key, value in Fibo2Caption._clean(caption).items()
            if key not in scores
        }
        caption = {key: value for key, value in caption.items() if value != {}}
        # A caption of exactly a short_description and a text_render list keeps both, the list even when it is empty:
        # it then says the image has no text
        if keys == {"short_description", "text_render"}:
            caption = {
                "short_description": caption.get("short_description") or "",
                "text_render": caption.get("text_render") or [],
            }
        return caption or None, instruction

    @staticmethod
    def generation_prompt(prompt: str) -> str:
        # A JSON caption in its trained form, serialized compactly; other prompts stay as they are
        caption, _ = Fibo2Caption.parse(prompt)
        return prompt if isinstance(caption, str) else Fibo2Caption.dumps(caption or {})

    @staticmethod
    def edit_prompt(prompt: str, num_images: int, text_images: bool = False) -> str:
        # One JSON object: a vision marker per image, the caption of the result when the prompt has one, then the
        # instruction. With images of the caption's text, each text entry gets the marker of its image.
        caption, instruction = Fibo2Caption.parse(prompt)
        if instruction is None:
            caption, instruction = None, prompt
        if text_images and isinstance(caption, dict):
            Fibo2Caption.mark_text_images(caption, VISION_MARKER)
        if num_images == 1:
            edit = {"image": VISION_MARKER}
        else:
            edit = {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        return Fibo2Caption.dumps(edit)

    @staticmethod
    def texts(prompt: str) -> list[str]:
        # The text_render texts of a prompt's caption, in order: one image is drawn for each
        caption, _ = Fibo2Caption.parse(prompt)
        entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
        return [entry["text"] for entry in entries if Fibo2Caption._has_text(entry)]

    @staticmethod
    def mark_text_images(caption: dict, marker: str) -> dict:
        # Gives each text_render entry that has text an "image_tokens" entry: the vision marker of the image of its
        # text, or "" where the image is not shown
        for entry in caption.get("text_render") or []:
            if Fibo2Caption._has_text(entry):
                entry["image_tokens"] = marker
        return caption

    @staticmethod
    def dumps(value) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _has_text(entry) -> bool:
        return isinstance(entry, dict) and isinstance(entry.get("text"), str) and bool(entry["text"])

    @staticmethod
    def _clean(value):
        if isinstance(value, dict):
            value = {k: Fibo2Caption._clean(v) for k, v in value.items()}
            return {k: v for k, v in value.items() if not Fibo2Caption._empty(v)}
        if isinstance(value, list):
            return [v for v in map(Fibo2Caption._clean, value) if not Fibo2Caption._empty(v)]
        return value

    @staticmethod
    def _empty(value) -> bool:
        if isinstance(value, float):
            return math.isnan(value)
        return value is None or (isinstance(value, (str, dict, list)) and not value)
