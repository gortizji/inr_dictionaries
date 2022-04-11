import jax
import jax.numpy as np
import neural_tangents as nt
from flax.core.frozen_dict import freeze
from scipy.sparse.linalg import eigsh


def get_ntk_fn(apply_fn, variables, batch_size):
    def apply_params_fn(params, x):
        if "batch_stats" in variables.keys():
            model_state, _ = variables.pop("params")
            apply_vars = make_variables(params, model_state)
            logits = apply_fn(apply_vars, x, train=False)
        else:
            logits = apply_fn(params, x)

        return logits

    kernel_fn = nt.batch(
        nt.empirical_kernel_fn(apply_params_fn, vmap_axes=0, implementation=2, trace_axes=()),
        batch_size=batch_size,
        device_count=0,
        store_on_device=False,
    )

    def expanded_kernel_fn(data1, data2, kernel_type, params):
        K = kernel_fn(data1, data2, kernel_type, params)
        return flatten_kernel(K)

    return expanded_kernel_fn


def ntk_eigendecomposition(apply_fn, variables, data, batch_size):
    kernel_fn = get_ntk_fn(apply_fn, variables, batch_size)

    if "params" in variables:
        ntk_matrix = kernel_fn(data, None, "ntk", variables["params"])
    else:
        ntk_matrix = kernel_fn(data, None, "ntk", variables)

    eigvals, eigvecs = eigsh(jax.device_get(ntk_matrix), k=2000)
    eigvals = np.flipud(eigvals)
    eigvecs = np.flipud(eigvecs.T)

    return eigvals, eigvecs, ntk_matrix


def make_variables(params, model_state):
    return freeze({"params": params, **model_state})


def flatten_kernel(K):
    return K.transpose([0, 2, 1, 3]).reshape([K.shape[0] * K.shape[2], K.shape[1] * K.shape[3]])
