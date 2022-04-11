import jax.numpy as jnp
import numpy as onp
import numpy as np
from sklearn.utils import shuffle


def crop_from_right(img, size):
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = size / 2
    r = int(r)
    cropped_image = img[c[0] - r : c[0] + r, img.shape[1] - 2 * r - 50 : img.shape[1] - 50]
    return cropped_image


def image_to_dataset(img, randomized_training_set=False, rand_state=0, rand_train_ratio=0.8):
    # Create input pixel coordinates in the unit square
    coords = np.linspace(-1, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)

    if randomized_training_set:
        x_perm, img_perm = shuffle(x_test, img)
        test_data = [x_perm, img_perm]
        c = int(jnp.sqrt(img.shape[0] * img.shape[1] * rand_train_ratio))
        train_data = [x_perm[0:c, 0:c], img_perm[0:c, 0:c]]

    else:
        test_data = [x_test, img]
        train_data = [x_test[::2, ::2], img[::2, ::2]]

    return train_data, test_data


def crop(img, size):
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = size / 2
    r = int(r)
    cropped_image = img[c[0] - r : c[0] + r, c[1] - r : c[1] + r]
    return cropped_image


def grayscale(img):
    if len(img.shape) >= 3:
        if img.shape[-1] == 3:
            gry_img = onp.matmul(img[..., :3], [0.299, 0.587, 0.114])
        else:
            gry_img = img

    else:
        gry_img = img

    return np.expand_dims(gry_img, -1)
