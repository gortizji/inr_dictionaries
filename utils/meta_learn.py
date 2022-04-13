import jax.numpy as jnp
import tensorflow_datasets as tfds
from skimage.transform import resize

from utils.img_processing import grayscale

DEFAULT_RESOLUTION = 64
x1 = jnp.linspace(0, 1, DEFAULT_RESOLUTION + 1)[:-1]
DEFAULT_GRID = jnp.stack(jnp.meshgrid(x1, x1, indexing="ij"), axis=-1)[None, ...]

CELEBA_BUILDER = tfds.builder("celeb_a", data_dir="gs://celeb_a_dataset/", version="2.0.0")
CELEBA_BUILDER.download_and_prepare()


def process_example(example, RES):
    # crop and resize the images
    # also turn them grayscale to make NTK computation possible on the trained network.
    RES_first = 178  # we will crop a square part from the center of the image with this resolution
    image = jnp.float32(example["image"]) / 255
    image = grayscale(image)
    image = image[
        :,
        image.shape[1] // 2 - RES_first // 2 : image.shape[1] // 2 + RES_first // 2,
        image.shape[2] // 2 - RES_first // 2 : image.shape[2] // 2 + RES_first // 2,
    ]
    image = resize(image, (image.shape[0], RES, RES), anti_aliasing=True)
    return image
