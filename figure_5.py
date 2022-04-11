import pickle

import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random

from models.models_haiku import FFN, MLP, SIREN
from utils.graphics import FOURIER_CMAP
from utils.meta_learn import DEFAULT_GRID, DEFAULT_RESOLUTION
from utils.ntk import ntk_eigendecomposition


def show_ntk_eigval_eigvec(
    model,
    params,
    data,
    batch_size,
    image_size,
    plot_eigvecs=range(10),
    cmap="gray",
    savefig=False,
    keyword="",
):

    eigvals, eigvecs, ntk_matrix = ntk_eigendecomposition(model.apply, params, data, batch_size)

    plt.figure()
    plt.plot(eigvals)
    sns.despine()
    if savefig:
        plt.savefig("figures/figure_5/" + keyword + "eigvals" + ".pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(eigvals / jnp.max(eigvals))
    sns.despine()
    if savefig:
        plt.savefig(
            "figures/figure_5/" + keyword + "eigvals_normalized" + ".pdf",
            bbox_inches="tight",
        )
    plt.close()

    for i in plot_eigvecs:
        plt.figure()

        v_i = eigvecs[i, :]
        v_i = jnp.reshape(v_i, [image_size, image_size])
        plt.imshow(v_i, cmap=cmap)
        plt.axis("off")
        if savefig:
            plt.savefig(
                "figures/figure_5/" + keyword + "eigvec" + str(i) + ".pdf",
                bbox_inches="tight",
            )

        plt.close()

    return eigvals, eigvecs, ntk_matrix


if __name__ == "__main__":
    BATCH_SIZE = 128
    DEFAULT_GRID = jnp.reshape(DEFAULT_GRID, [-1, 2])

    # Build models to compare
    model_SIREN = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=30, width=256, hidden_w0=30, depth=5)(x))
    )
    params_SIREN = model_SIREN.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_MLP = hk.without_apply_rng(hk.transform(lambda x: MLP(width=256, depth=5)(x)))
    params_mlp = model_MLP.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_5 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=5, width=256, hidden_w0=30, depth=5)(x))
    )
    params_SIREN_5 = model_SIREN_5.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_100 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=100, width=256, hidden_w0=30, depth=5)(x))
    )
    params_SIREN_100 = model_SIREN_100.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_FFN_1 = hk.without_apply_rng(hk.transform(lambda x: FFN(sigma=1, width=256, depth=5)(x)))
    params_FFN_1 = model_FFN_1.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_FFN_10 = hk.without_apply_rng(hk.transform(lambda x: FFN(sigma=1, width=256, depth=5)(x)))
    params_FFN_10 = model_FFN_10.init(random.PRNGKey(0), jnp.ones((1, 2)))

    with open("maml_celebA_5000.pickle", "rb") as handle:
        params_meta = pickle.load(handle)

    # Plot eigenvectors
    print("Computing and plotting NTK eigenvectors of SIREN (meta)...")
    eigvals_meta, eigvecs_meta, ntk_matrix_meta = show_ntk_eigval_eigvec(
        model_SIREN,
        params_meta,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="meta_maml",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )

    print("Computing and plotting NTK eigenvectors of MLP...")
    eigvals_mlp, eigvecs_mlp, ntk_matrix_mlp = show_ntk_eigval_eigvec(
        model_MLP,
        params_mlp,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="relu_mlp_",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )

    print("Computing and plotting NTK eigenvectors of SIREN-5...")
    eigvals_5, eigvecs_5, ntk_matrix_5 = show_ntk_eigval_eigvec(
        model_SIREN_5,
        params_SIREN_5,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="siren_5_",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )

    print("Computing and plotting NTK eigenvectors of SIREN-30...")
    eigvals_30, eigvecs_30, ntk_matrix_30 = show_ntk_eigval_eigvec(
        model_SIREN,
        params_SIREN,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="siren_30_",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )

    print("Computing and plotting NTK eigenvectors of SIREN-100...")
    eigvals_100, eigvecs_100, ntk_matrix_100 = show_ntk_eigval_eigvec(
        model_SIREN_100,
        params_SIREN_100,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="siren_100_",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )

    print("Computing and plotting NTK eigenvectors of FFN-10...")
    eigvals_ffn_10, eigvecs_ffn_10, ntk_matrix_ffn_10 = show_ntk_eigval_eigvec(
        model_FFN_10,
        params_FFN_10,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="ffn_10_",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )

    print("Computing and plotting NTK eigenvectors of FFN-1...")
    eigvals_ffn_1, eigvecs_ffn_1, ntk_matrix_ffn_1 = show_ntk_eigval_eigvec(
        model_FFN_1,
        params_FFN_1,
        cmap=FOURIER_CMAP,
        savefig=True,
        keyword="ffn_1_",
        data=DEFAULT_GRID,
        image_size=DEFAULT_RESOLUTION,
        batch_size=BATCH_SIZE,
    )
