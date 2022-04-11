import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.img_processing import grayscale

FOURIER_CMAP = sns.cubehelix_palette(8, start=0.5, rot=-0.75, as_cmap=True, reverse=True)


def plot_fourier_tranform(img):
    gray_img = grayscale(img)
    fft_img = np.abs(np.fft.fftshift(np.fft.fft2(gray_img)))
    plt.imshow(20 * np.log(np.abs(fft_img)), cmap=FOURIER_CMAP)
    plt.axis("off")
    plt.colorbar()


def plot_error(
    outputs,
):
    # Plot train/test error curves
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(outputs["xs"], outputs["train_psnrs"])
    plt.title("Train error")
    plt.ylabel("PSNR")
    plt.xlabel("Training iter")
    plt.subplot(122)
    plt.plot(outputs["xs"], outputs["test_psnrs"])
    plt.title("Test error")
    plt.ylabel("PSNR")
    plt.xlabel("Training iter")
    plt.show()
