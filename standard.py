import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import optimizers
from tqdm import tqdm


def fit_image(
    model,
    train_data,
    test_data,
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
    input_dim=2,
):
    rand_key = jax.random.PRNGKey(rand_state)
    key_i = jax.random.split(rand_key, 2 * iters)

    @jax.jit
    def model_pred(params, x):
        return model.apply(params, x)

    @jax.jit
    def model_loss(params, x, y):
        return 0.5 * jnp.mean((model_pred(params, x) - y) ** 2)

    @jax.jit
    def model_psnr(params, x, y):
        return -10 * jnp.log10(2.0 * model_loss(params, x, y))

    @jax.jit
    def model_grad_loss(params, x, y):
        return jax.grad(model_loss)(params, x, y)

    if optimizer_type == "adam":
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_update = jax.jit(opt_update)
    elif optimizer_type == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
        opt_update = jax.jit(opt_update)
    else:
        opt_init, opt_update, get_params = optimizer

    if initial_params is None:
        params = model.init(rand_key, jnp.zeros(input_dim, jnp.float32))
    else:
        params = initial_params

        if last_layer_rand_init:
            params = unfreeze(params)
            rand_params = model.init(rand_key, jnp.zeros(input_dim, jnp.float32))
            num_layers = len(params["params"])
            for idx, layer in enumerate(params["params"]):
                if idx == num_layers - 1:
                    print(layer)
                    params["params"][layer]["bias"] = rand_params["params"][layer]["bias"]
                    params["params"][layer]["kernel"] = rand_params["params"][layer]["kernel"]

        params = freeze(params)

    if opt_state is None:
        opt_state = opt_init(params)

    train_psnrs = []
    train_loss = []
    test_psnrs = []
    pred_imgs = []
    xs = []
    param_state = []
    grads = []

    train_psnrs.append(model_psnr(get_params(opt_state), *train_data))
    test_psnrs.append(model_psnr(get_params(opt_state), *test_data))
    xs.append(start_iter - 1)

    for i in tqdm(range(start_iter, start_iter + iters)):
        if batch_size is None:
            grad_i = model_grad_loss(get_params(opt_state), *train_data)
        else:
            coord_0 = jax.random.randint(
                key_i[i - start_iter], (batch_size,), 0, train_data[0].shape[0]
            )
            coord_1 = jax.random.randint(
                key_i[-i + start_iter], (batch_size,), 0, train_data[0].shape[1]
            )
            batch_i = [
                train_data[0][coord_0, coord_1, :],
                train_data[1][coord_0, coord_1, :],
            ]

            grad_i = model_grad_loss(get_params(opt_state), *batch_i)

        opt_state = opt_update(i, grad_i, opt_state)

        if i % log_every == 0:
            param_state.append(get_params(opt_state))
            grads.append(grad_i)
            train_psnrs.append(model_psnr(get_params(opt_state), *train_data))
            test_psnrs.append(model_psnr(get_params(opt_state), *test_data))
            train_loss.append(model_loss(get_params(opt_state), *train_data))
            pred_imgs.append(model_pred(get_params(opt_state), test_data[0]))
            xs.append(i)

        outputs = {
            "state": get_params(opt_state),
            "train_psnrs": train_psnrs,
            "test_psnrs": test_psnrs,
            "train_loss": train_loss,
            "pred_imgs": jnp.stack(pred_imgs),
            "xs": xs,
            "param_states": param_state,
            "gradient": grads,
            "optimizer": (opt_init, opt_update, get_params),
            "opt_state": opt_state,
        }

    return outputs, get_params
