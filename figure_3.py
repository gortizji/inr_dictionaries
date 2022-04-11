import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.models_flax import SIREN
from train.standard import fit_image


def multi_tone_fitting(
    model,
    num_samples,
    k_values=[],
    learning_rate=1e-4,
    iters=2000,
    test_train_ratio=2,
    rand_state=0,
):
    T_s = 1 / num_samples
    coords = np.linspace(0, 1, num_samples, endpoint=False)
    x_test = np.expand_dims(coords, axis=-1)

    freq = np.fft.fftfreq(num_samples, d=T_s)
    fun_k = generate_signal_with_components_equal_amplitude(
        comp_cos=[], comp_sin=k_values, num_samples=num_samples
    )

    data_whole = [x_test, np.expand_dims(fun_k, axis=-1)]
    train_set = [
        x_test[::test_train_ratio, :],
        np.expand_dims(fun_k, axis=-1)[::test_train_ratio, :],
    ]

    train_data = train_set
    test_data = data_whole

    @jax.jit
    def model_pred(params, x):
        return model.apply(params, x)

    outputs, get_params = fit_image(
        model,
        train_data,
        test_data,
        "adam",
        batch_size=None,
        learning_rate=learning_rate,
        iters=iters,
        rand_state=rand_state,
        input_dim=1,
    )

    # Show final network outputs
    rec = outputs["pred_imgs"][-1].flatten()
    freq = jnp.fft.fftfreq(len(rec), d=1 / len(rec))
    fft_rec = 20 * jnp.log10(jnp.abs(jnp.fft.fft(rec)))
    rec_test_ft = jnp.fft.fftshift(fft_rec)
    test_freqs = jnp.fft.fftshift(freq)
    gt_test_ft = 20 * jnp.log10(np.fft.fftshift(jnp.abs(jnp.fft.fft(fun_k))))

    rec = model_pred(get_params(outputs["opt_state"]), train_data[0])
    rec = rec.flatten()
    freq = jnp.fft.fftfreq(len(rec), d=1 / len(rec))
    fft_rec = 20 * jnp.log10(jnp.abs(jnp.fft.fft(rec)))
    train_freqs = jnp.fft.fftshift(freq)
    rec_train_ft = jnp.fft.fftshift(fft_rec)
    gt_train_ft = 20 * jnp.log10(
        np.fft.fftshift(jnp.abs(jnp.fft.fft(fun_k[::test_train_ratio].flatten())))
    )
    return gt_train_ft, rec_train_ft, train_freqs, gt_test_ft, rec_test_ft, test_freqs


def generate_signal_with_components_equal_amplitude(
    comp_cos, comp_sin=[], num_samples=256, random_phase=True, random_amplitude=False
):
    coords = np.linspace(0, 1, num_samples, endpoint=False)
    fun_k = np.zeros(num_samples)
    for k in comp_cos:
        if random_amplitude:
            a_k = np.random.uniform(0, 1)
        else:
            a_k = 1
        if random_phase:
            phase = np.random.uniform(0, 2 * np.pi)
        else:
            phase = 0
        fun_k += a_k * jnp.cos(2 * jnp.pi * k * coords + phase)
    for k in comp_sin:
        if random_amplitude:
            a_k = np.random.uniform(0, 1)
        else:
            a_k = 1
        if random_phase:
            phase = np.random.uniform(0, 2 * np.pi)
        else:
            phase = 0
        fun_k += a_k * jnp.sin(2 * jnp.pi * k * coords + phase)
    return fun_k


if __name__ == "__main__":
    K_COMPONENTS = [23]
    NETWORK_SIZE = [128, 128, 1]
    NUM_SAMPLES = 256
    ITERS = 2000
    LEARNING_RATE = 1e-4

    # Part I
    print("Fitting sinusoid with SIREN-30")
    (
        gt_train_ft,
        rec_train_ft,
        train_freqs,
        gt_test_ft,
        rec_test_ft,
        test_freqs,
    ) = multi_tone_fitting(
        SIREN(features=NETWORK_SIZE, first_omega_0=30, hidden_omega_0=30, input_dim=1),
        num_samples=NUM_SAMPLES,
        k_values=K_COMPONENTS,
        iters=ITERS,
        learning_rate=LEARNING_RATE,
        test_train_ratio=2,
    )
    sns.set_context("paper", font_scale=3)

    fig, ax = plt.subplots()
    ax.plot(train_freqs, rec_train_ft, label="Rec", linewidth=3)
    ax.plot(train_freqs, gt_train_ft, label="GT", linestyle="dashed", linewidth=3)
    sns.despine()
    ax.annotate(
        "$f$ (Hz)",
        xy=(1.01, 0.03),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "Magnitude Spectrum (dB)",
        xy=(-0.25, 1.1),
        xytext=(-15, 2),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "$f=23$",
        xy=(0.60, 0.87),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    ax.annotate(
        r"$f=\!\!-\!23$",
        xy=(0.25, 0.87),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    plt.ylim([-130, 90])
    plt.savefig("figures/figure_3/train_grid_merged_30.pdf", bbox_inches="tight")
    sns.set_context("paper", font_scale=3)

    fig, ax = plt.subplots()
    ax.plot(test_freqs, rec_test_ft, label="$f_{\\theta}(r)$", linewidth=3)
    ax.plot(test_freqs, gt_test_ft, label="$g(r)$", linestyle="dashed", linewidth=3)
    plt.ylim(-20, 60)
    ax.annotate(
        "$f$ (Hz)",
        xy=(1.01, 0.03),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "Magnitude Spectrum (dB)",
        xy=(-0.25, 1.1),
        xytext=(-15, 2),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "$f=23$",
        xy=(0.55, 0.92),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    ax.annotate(
        r"$f=\!\!-\!23$",
        xy=(0.23, 0.92),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    sns.despine()
    ax.set_yticks([-30, 0, 30])
    plt.legend(loc=(0.8, 0.7))
    plt.savefig("figures/figure_3/test_grid_merged_30.pdf", bbox_inches="tight")

    # Part II
    print("Fitting sinusoid with SIREN-300")
    (
        gt_train_ft,
        rec_train_ft,
        train_freqs,
        gt_test_ft,
        rec_test_ft,
        test_freqs,
    ) = multi_tone_fitting(
        SIREN(features=NETWORK_SIZE, first_omega_0=300, hidden_omega_0=30, input_dim=1),
        num_samples=NUM_SAMPLES,
        k_values=K_COMPONENTS,
        iters=ITERS,
        learning_rate=LEARNING_RATE,
        test_train_ratio=2,
    )

    sns.set_context("paper", font_scale=3)
    fig, ax = plt.subplots()
    ax.plot(test_freqs, rec_test_ft, label="$f_{\\theta}(r)$", linewidth=3)
    ax.plot(test_freqs, gt_test_ft, label="$g(r)$", linestyle="dashed", linewidth=3)
    plt.vlines(105, -20, 35, linestyles="dashed", color="yellowgreen", linewidth=2)
    plt.vlines(-105, -20, 35, linestyles="dashed", color="yellowgreen", linewidth=2)
    plt.ylim(-50, 70)
    ax.annotate(
        "$f$ (Hz)",
        xy=(1.01, 0.03),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "Magnitude Spectrum (dB)",
        xy=(-0.25, 1.1),
        xytext=(-15, 2),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "$f=23$",
        xy=(0.55, 0.9),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    ax.annotate(
        r"$f=\!\!-\!23$",
        xy=(0.2, 0.9),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    ax.annotate(
        "$f=105$",
        xy=(0.80, 0.75),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    ax.annotate(
        r"$f=\!\!-\!105$",
        xy=(0.05, 0.75),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    sns.despine()
    ax.set_yticks([0, 20, 40])
    plt.savefig("figures/figure_3/test_grid_merged_128_300.pdf", bbox_inches="tight")

    sns.set_context("paper", font_scale=3)
    fig, ax = plt.subplots()
    ax.plot(train_freqs, rec_train_ft, label="Rec", linewidth=3)
    ax.plot(train_freqs, gt_train_ft, label="GT", linestyle="dashed", linewidth=3)
    sns.despine()
    ax.annotate(
        "$f$ (Hz)",
        xy=(1.01, 0.03),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    ax.annotate(
        "Magnitude Spectrum (dB)",
        xy=(-0.25, 1.1),
        xytext=(-15, 2),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=25,
    )
    plt.ylim(-80, 80)
    ax.annotate(
        "$f=23$",
        xy=(0.6, 0.85),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    ax.annotate(
        r"$f=\!\!-\!23$",
        xy=(0.25, 0.85),
        ha="left",
        va="top",
        xycoords="axes fraction",
        textcoords="offset points",
        color="darkgreen",
        fontsize=25,
    )
    plt.savefig("figures/figure_3/train_grid_merged_128_300.pdf", bbox_inches="tight")
