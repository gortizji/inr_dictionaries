import pickle

import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow_datasets as tfds
from jax import random
from tqdm import tqdm

from models.models_haiku import MLP, SIREN
from utils.graphics import FOURIER_CMAP
from utils.meta_learn import CELEBA_BUILDER, DEFAULT_GRID, DEFAULT_RESOLUTION, process_example
from utils.ntk import ntk_eigendecomposition


def energy_eigval_treshold(eigvecs, eigvals, num_examples, ds_test):
    # given eigvecs and eigvals project the num_examples many test images to eigenvectors
    # returns the mean and std of the percent energy covered by eigevectors up to that index
    # and the percent energy covered by eigenvectors whose eigenvalues are greater than max_eigval*treshold
    energies = np.zeros((num_examples, len(eigvals)))
    for i, example in tqdm(enumerate(tfds.as_numpy(ds_test))):
        test_img = process_example(example, DEFAULT_RESOLUTION)
        test_img = jnp.squeeze(test_img)

        test_img = test_img - jnp.mean(test_img)

        vec_img = test_img.flatten()
        img_ntk_eigvec_basis_rep = eigvecs @ vec_img
        norm_img = jnp.linalg.norm(vec_img)

        energy_in_comp = jnp.square(jnp.abs(img_ntk_eigvec_basis_rep)) / jnp.square(norm_img)

        percent_energy = np.zeros_like(energy_in_comp)
        for j, val in enumerate(energy_in_comp):
            percent_energy[j] = val + percent_energy[j - 1]

        energies[i, :] = percent_energy

    random_energy_mean = jnp.mean(energies, axis=0)
    random_energy_std = jnp.std(energies, axis=0)

    eigvals_nor = eigvals / jnp.max(eigvals)
    num_eigvals_treshold = jnp.sum(eigvals_nor >= 0.1)
    energy_covered = random_energy_mean[num_eigvals_treshold - 1]

    return random_energy_mean, random_energy_std, energy_covered


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_EXAMPLES = 100
    DEFAULT_GRID = jnp.reshape(DEFAULT_GRID, [-1, 2])

    ds_test = CELEBA_BUILDER.as_dataset(
        split="test", as_supervised=False, shuffle_files=False, batch_size=1
    )
    ds_test = ds_test.take(NUM_EXAMPLES)

    # Build models to compare
    with open("maml_celebA_5000.pickle", "rb") as handle:
        params_meta = pickle.load(handle)

    model_MLP = hk.without_apply_rng(hk.transform(lambda x: MLP(width=256, depth=5)(x)))
    params_mlp = model_MLP.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=30, width=256, hidden_w0=30, depth=5)(x))
    )
    params = model_SIREN.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_5 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=5, width=256, hidden_w0=30, depth=5)(x))
    )
    params_5 = model_SIREN_5.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_10 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=10, width=256, hidden_w0=30, depth=5)(x))
    )
    params_10 = model_SIREN_10.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_20 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=20, width=256, hidden_w0=30, depth=5)(x))
    )
    params_20 = model_SIREN_20.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_40 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=40, width=256, hidden_w0=30, depth=5)(x))
    )
    params_40 = model_SIREN_40.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_60 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=60, width=256, hidden_w0=30, depth=5)(x))
    )
    params_60 = model_SIREN_60.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_80 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=80, width=256, hidden_w0=30, depth=5)(x))
    )
    params_80 = model_SIREN_80.init(random.PRNGKey(0), jnp.ones((1, 2)))

    model_SIREN_100 = hk.without_apply_rng(
        hk.transform(lambda x: SIREN(w0=100, width=256, hidden_w0=30, depth=5)(x))
    )
    params_100 = model_SIREN_100.init(random.PRNGKey(0), jnp.ones((1, 2)))

    # Compute their NTK eigenvectors

    print("NTK eigendecomposition of SIREN-(meta)...")
    eigvals_meta, eigvecs_meta, ntk_matrix_meta = ntk_eigendecomposition(
        model_SIREN.apply, params_meta, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of MLP...")
    eigvals_mlp, eigvecs_mlp, ntk_matrix_mlp = ntk_eigendecomposition(
        model_MLP.apply, params_mlp, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-5...")
    eigvals_5, eigvecs_5, ntk_matrix_5 = ntk_eigendecomposition(
        model_SIREN_5.apply, params_5, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-10...")
    eigvals_10, eigvecs_10, ntk_matrix_10 = ntk_eigendecomposition(
        model_SIREN_10.apply, params_10, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-20...")
    eigvals_20, eigvecs_20, ntk_matrix_20 = ntk_eigendecomposition(
        model_SIREN_20.apply, params_20, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-40...")
    eigvals_40, eigvecs_40, ntk_matrix_40 = ntk_eigendecomposition(
        model_SIREN_40.apply, params_40, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-60...")
    eigvals_60, eigvecs_60, ntk_matrix_60 = ntk_eigendecomposition(
        model_SIREN_60.apply, params_60, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-80...")
    eigvals_80, eigvecs_80, ntk_matrix_80 = ntk_eigendecomposition(
        model_SIREN_80.apply, params_80, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    print("NTK eigendecomposition of SIREN-100...")
    eigvals_100, eigvecs_100, ntk_matrix_100 = ntk_eigendecomposition(
        model_SIREN_100.apply, params_100, data=DEFAULT_GRID, batch_size=BATCH_SIZE
    )

    # Plot the energy-eigenvalue plot
    print("Projecting on SIREN-(meta)...")
    (
        meta_energy_mean_demean,
        meta_energy_std_demean,
        meta_energy_covered_demean,
    ) = energy_eigval_treshold(
        eigvecs_meta, eigvals_meta, num_examples=NUM_EXAMPLES, ds_test=ds_test
    )

    print("Projecting on MLP...")
    (
        random_energy_mean_demean_mlp,
        random_energy_std_demean_mlp,
        random_energy_covered_demean_mlp,
    ) = energy_eigval_treshold(eigvecs_mlp, eigvals_mlp, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-5...")
    (
        random_energy_mean_demean_5,
        random_energy_std_demean_5,
        random_energy_covered_demean_5,
    ) = energy_eigval_treshold(eigvecs_5, eigvals_5, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-10...")
    (
        random_energy_mean_demean_10,
        random_energy_std_demean_10,
        random_energy_covered_demean_10,
    ) = energy_eigval_treshold(eigvecs_10, eigvals_10, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-20...")
    (
        random_energy_mean_demean_20,
        random_energy_std_demean_20,
        random_energy_covered_demean_20,
    ) = energy_eigval_treshold(eigvecs_20, eigvals_20, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-40...")
    (
        random_energy_mean_demean_40,
        random_energy_std_demean_40,
        random_energy_covered_demean_40,
    ) = energy_eigval_treshold(eigvecs_40, eigvals_40, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-60...")
    (
        random_energy_mean_demean_60,
        random_energy_std_demean_60,
        random_energy_covered_demean_60,
    ) = energy_eigval_treshold(eigvecs_60, eigvals_60, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-80...")
    (
        random_energy_mean_demean_80,
        random_energy_std_demean_80,
        random_energy_covered_demean_80,
    ) = energy_eigval_treshold(eigvecs_80, eigvals_80, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    print("Projecting on SIREN-100...")
    (
        random_energy_mean_demean_100,
        random_energy_std_demean_100,
        random_energy_covered_demean_100,
    ) = energy_eigval_treshold(eigvecs_100, eigvals_100, num_examples=NUM_EXAMPLES, ds_test=ds_test)

    colors = FOURIER_CMAP(jnp.linspace(0, 1, 7))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogx(
        eigvals_mlp / jnp.max(eigvals_mlp),
        random_energy_mean_demean_mlp,
        label="ReLU-MLP",
        linewidth=2,
        linestyle="dashed",
        color="slategrey",
    )
    ax.semilogx(
        eigvals_5 / jnp.max(eigvals_5),
        random_energy_mean_demean_5,
        label="SIREN-5",
        linewidth=2,
        color=colors[0],
    )
    ax.semilogx(
        eigvals_10 / jnp.max(eigvals_10),
        random_energy_mean_demean_10,
        label="SIREN-10",
        linewidth=2,
        color=colors[1],
    )
    ax.semilogx(
        eigvals_20 / jnp.max(eigvals_20),
        random_energy_mean_demean_20,
        label="SIREN-20",
        linewidth=2,
        color=colors[2],
    )
    ax.semilogx(
        eigvals_40 / jnp.max(eigvals_40),
        random_energy_mean_demean_40,
        label="SIREN-40",
        linewidth=2,
        color=colors[3],
    )
    ax.semilogx(
        eigvals_60 / jnp.max(eigvals_60),
        random_energy_mean_demean_60,
        label="SIREN-60",
        linewidth=2,
        color=colors[4],
    )
    ax.semilogx(
        eigvals_80 / jnp.max(eigvals_80),
        random_energy_mean_demean_80,
        label="SIREN-80",
        linewidth=2,
        color=colors[5],
    )
    ax.semilogx(
        eigvals_100 / jnp.max(eigvals_100),
        random_energy_mean_demean_100,
        label="SIREN-100",
        linewidth=2,
        color=colors[6],
    )
    ax.semilogx(
        eigvals_meta / jnp.max(eigvals_meta),
        meta_energy_mean_demean,
        label="SIREN (Meta)",
        linewidth=2,
        color="darkred",
    )

    plt.xlim(1, 1e-10)
    sns.despine()

    lgd = ax.legend(loc=(1.01, 0.01))

    plt.savefig(
        "figures/figure_4/fig4.pdf",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
