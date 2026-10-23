import mlx.core as mx
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.image_util import ImageUtil


class Fibo2LatentCreator:
    # Fibo 2.0 works on the FLUX.2 VAE latents normalized with the VAE's BatchNorm statistics, patchified to
    # (1, 128, H / 16, W / 16): the transformer's 2x2 patches are those 128 channels.
    LATENT_CHANNELS = 128
    SPATIAL_SCALE = 16

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        # The reference draws the starting noise in float32 and keeps the latents in float32 between steps
        Fibo2LatentCreator.check_size(height=height, width=width)
        shape = (1, Fibo2LatentCreator.LATENT_CHANNELS, height // Fibo2LatentCreator.SPATIAL_SCALE, width // Fibo2LatentCreator.SPATIAL_SCALE)  # fmt: off
        return mx.random.normal(shape=shape, key=mx.random.key(seed), dtype=mx.float32)

    @staticmethod
    def check_size(height: int, width: int) -> None:
        scale = Fibo2LatentCreator.SPATIAL_SCALE
        if height <= 0 or width <= 0 or height % scale or width % scale:
            raise ValueError(f"Fibo 2.0 needs a width and height that are positive multiples of 16, got {width}x{height}.")  # fmt: off

    @staticmethod
    def encode_image(vae: Flux2VAE, image: Image.Image, tiling_config: TilingConfig | None = None) -> mx.array:
        # An image to edit, as the transformer reads it: encoded (the latent distribution's mode), patchified and
        # normalized like the image being generated
        encoded = VAEUtil.encode(vae=vae, image=ImageUtil.to_array(image).astype(ModelConfig.precision), tiling_config=tiling_config)  # fmt: off
        if encoded.ndim == 5:
            encoded = encoded[:, :, 0]
        patchified = Flux2LatentCreator.patchify_latents(encoded).astype(mx.float32)
        bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps)
        return ((patchified - bn_mean) / bn_std).astype(ModelConfig.precision)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        # The latents already are the packed 128-channel FLUX.2 layout, which decode_packed_latents takes
        return latents
