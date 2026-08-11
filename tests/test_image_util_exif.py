import piexif
import PIL.Image
import pytest

from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exif_orientation import oriented_size
from mflux.utils.image_util import ImageUtil
from mflux.utils.scale_factor import ScaleFactor


def _save_with_orientation(path, orientation: int) -> None:
    # 4x2 landscape with a red left column so the rotation is observable in pixels.
    image = PIL.Image.new("RGB", (4, 2), color=(0, 0, 255))
    for y in range(2):
        image.putpixel((0, y), (255, 0, 0))
    exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}})
    image.save(path, format="JPEG", exif=exif, quality=100, subsampling=0)


@pytest.mark.fast
def test_load_image_applies_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    _save_with_orientation(path, orientation=6)

    loaded = ImageUtil.load_image(path)

    # Orientation 6 means the stored pixels need a 270-degree rotation to display upright,
    # so the 4x2 file must come back as 2x4 with the red column now the top row.
    assert loaded.size == (2, 4)
    assert loaded.getpixel((0, 0))[0] > 128
    assert loaded.getpixel((1, 0))[0] > 128
    assert loaded.getpixel((0, 3))[2] > 128


@pytest.mark.fast
def test_load_image_without_orientation_is_untouched(tmp_path):
    path = tmp_path / "plain.jpg"
    PIL.Image.new("RGB", (4, 2), color=(0, 0, 255)).save(path, format="JPEG")

    loaded = ImageUtil.load_image(path)

    assert loaded.size == (4, 2)


@pytest.mark.fast
def test_load_image_applies_orientation_on_in_memory_images(tmp_path):
    path = tmp_path / "rotated.jpg"
    _save_with_orientation(path, orientation=6)

    loaded = ImageUtil.load_image(PIL.Image.open(path))

    assert loaded.size == (2, 4)


@pytest.mark.fast
def test_oriented_size_matches_the_loaded_pixels(tmp_path):
    # Every site that reads a path has to describe the same picture. A size read that
    # ignores the tag puts dimensions ninety degrees away from the pixels the model gets.
    path = tmp_path / "rotated.jpg"
    _save_with_orientation(path, orientation=6)

    assert oriented_size(path) == ImageUtil.load_image(path).size

    plain = tmp_path / "plain.jpg"
    PIL.Image.new("RGB", (4, 2), color=(0, 0, 255)).save(plain, format="JPEG")
    assert oriented_size(plain) == (4, 2)


@pytest.mark.fast
def test_dimension_resolver_agrees_with_the_loader_on_rotated_files(tmp_path):
    # Big enough that the resolver's rounding to multiples of 16 leaves both axes non-zero.
    path = tmp_path / "rotated.jpg"
    image = PIL.Image.new("RGB", (400, 200), color=(0, 0, 255))
    exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}})
    image.save(path, format="JPEG", exif=exif)

    one = ScaleFactor(value=1.0)
    width, height = DimensionResolver.resolve(height=one, width=one, reference_image_path=path)
    loaded_width, loaded_height = ImageUtil.load_image(path).size

    # Both are portrait, or both are landscape: the resolver must not invert the target.
    assert (width < height) == (loaded_width < loaded_height)
