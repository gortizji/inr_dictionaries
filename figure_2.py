import imageio
import jax
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from models.models_flax import FFN
from train.standard import fit_image
from utils.graphics import plot_fourier_tranform
from utils.img_processing import crop_from_right, image_to_dataset


def plot_reconstructions(
    outputs,
    image_GT,
    save_phrase="",
):
    # Show final network outputs
    plt.figure(figsize=(12, 4))
    rec = outputs["pred_imgs"][-1]
    plt.imshow(rec)
    plt.axis("off")
    plt.savefig("figures/figure_2/rec_" + save_phrase + ".pdf", bbox_inches="tight")

    plt.figure()
    plt.imshow(image_GT)
    plt.axis("off")
    plt.savefig("figures/figure_2/gt_" + save_phrase + ".pdf", bbox_inches="tight")

    plt.figure()
    plot_fourier_tranform(rec)
    plt.savefig("figures/figure_2/rec_ft_" + save_phrase + ".pdf", bbox_inches="tight")

    plt.figure()
    plot_fourier_tranform(image_GT)
    plt.savefig("figures/figure_2/gt_ft_" + save_phrase + ".pdf", bbox_inches="tight")


def train_and_plot_image(
    model,
    train_data,
    test_data,
    image_GT,
    optimizer_type="adam",
    batch_size=None,
    start_iter=0,
    initial_params=None,
    optimizer=None,
    opt_state=None,
    last_layer_rand_init=False,
    log_every=25,
    iters=2000,
    learning_rate=1e-4,
    rand_state=0,
    save_phrase="",
):
    outputs, _ = fit_image(
        model,
        train_data,
        test_data,
        optimizer_type,
        batch_size,
        start_iter,
        initial_params,
        optimizer,
        opt_state,
        last_layer_rand_init,
        log_every,
        iters,
        learning_rate,
        rand_state,
    )

    plot_reconstructions(outputs, image_GT, save_phrase)
    return outputs


if __name__ == "__main__":
    # save GT image and create test/train data
    image_url = "https://i.imgur.com/OQnG76L.jpeg"
    img = imageio.imread(image_url)
    img = img / 255
    img = crop_from_right(img, 960)
    img = resize(img, (512, 512), anti_aliasing=True)

    # create a dataset out of that image
    _, img_data = image_to_dataset(img)

    print("Reconstructing with FFN (sigma=10)")
    outputs = train_and_plot_image(
        FFN(
            features=np.array([256, 256, 256, 3]),
            B=10 * jax.random.normal(jax.random.PRNGKey(7), (256, 2)),
        ),
        train_data=img_data,
        test_data=img_data,
        image_GT=img,
        iters=2000,
        save_phrase="rff_256",
    )

    print("Reconstructing with single frequency mapping (f0=1)")
    # single frequency mapping bbf-1
    outputs = train_and_plot_image(
        FFN(features=np.array([256, 256, 256, 3]), B=np.eye(2)),
        train_data=img_data,
        test_data=img_data,
        image_GT=img,
        iters=2000,
        save_phrase="bff_1",
    )

    print("Reconstructing with single frequency mapping (f0=05)")
    # single frequency mapping bff-05
    outputs = train_and_plot_image(
        FFN(features=np.array([256, 256, 256, 3]), B=0.5 * np.eye(2)),
        train_data=img_data,
        test_data=img_data,
        image_GT=img,
        iters=2000,
        save_phrase="bff_05",
    )
