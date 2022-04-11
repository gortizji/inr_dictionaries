import jax
import jax.numpy as np
import optax
import tensorflow_datasets as tfds
from jax import jit, random, vmap

from utils.meta_learn import DEFAULT_RESOLUTION, process_example


def meta_train(
    model,
    params,
    coords,
    ds,
    ds_val,
    batch_size,
    outer_lr=1e-5,
    inner_lr=0.01,
    inner_steps=2,
    meta_method="MAML",
    max_iters=5000,
):
    def outer_step(rng, image, coords, params, inner_steps, opt_inner):
        def loss_fn(params, rng_input):
            g = model.apply(params, coords)
            return mse_fn(g, image)

        image = np.reshape(image, (-1, 1))
        coords = np.reshape(coords, (-1, 2))
        opt_inner_state = opt_inner.init(params)
        loss = 0
        for _ in range(inner_steps):
            rng, rng_input = random.split(rng)
            loss, grad = jax.value_and_grad(loss_fn)(params, rng_input)

            updates, opt_inner_state = opt_inner.update(grad, opt_inner_state)
            params = optax.apply_updates(params, updates)
        return rng, params, loss

    outer_step_batch = vmap(outer_step, in_axes=[0, 0, None, None, None, None])

    def update_model(rng, params, opt_state, image, coords, inner_steps, opt_inner, reptile=False):

        if reptile:
            rng, new_params, loss = outer_step_batch(
                rng, image, coords, params, inner_steps, opt_inner
            )
            rng, loss = rng[0], np.mean(loss)
            new_params = jax.tree_map(lambda x: np.mean(x, axis=0), new_params)

            def calc_grad(params, new_params):
                return params - new_params

            model_grad = jax.tree_multimap(calc_grad, params, new_params)
        else:

            def loss_model(params, rng):
                rng = random.split(rng, batch_size)
                rng, new_params, loss = outer_step_batch(
                    rng, image, coords, params, inner_steps, opt_inner
                )
                g = vmap(model.apply, in_axes=[0, None])(new_params, coords)
                return mse_fn(g[:, 0, ...], image), rng[0]

            (loss, rng), model_grad = jax.value_and_grad(loss_model, argnums=0, has_aux=True)(
                params, rng
            )

        updates, opt_state = opt_outer.update(model_grad, opt_state)
        params = optax.apply_updates(params, updates)
        return rng, params, opt_state, loss

    opt_outer = optax.adam(outer_lr)
    opt_inner = optax.sgd(inner_lr)

    opt_state = opt_outer.init(params)

    mse_fn = jit(lambda x, y: np.mean((x - y) ** 2))
    psnr_fn = jit(lambda mse: -10 * np.log10(mse))

    train_psnrs = []
    train_psnr = []
    val_psnrs = []
    steps = []
    step = 0
    rng = random.PRNGKey(0)
    rng_test = random.PRNGKey(42)
    while step < max_iters:
        for example in tfds.as_numpy(ds):
            if step > max_iters:
                break

            image = process_example(example, DEFAULT_RESOLUTION)

            rng, params, opt_state, loss = update_model(
                rng,
                params,
                opt_state,
                image,
                coords,
                inner_steps,
                opt_inner,
                reptile=(meta_method == "REPTILE"),
            )
            train_psnr.append(psnr_fn(loss))

            if step % 200 == 0 and step != 0:
                train_psnrs.append(np.mean(np.array(train_psnr)))
                train_psnr = []
                val_psnr = []
                for val_example in tfds.as_numpy(ds_val):
                    val_img = process_example(val_example, DEFAULT_RESOLUTION)
                    _, params_test, loss = outer_step(
                        rng_test, val_img, coords, params, inner_steps, opt_inner
                    )
                    img = model.apply(params_test, coords)[0]
                    val_psnr.append(psnr_fn(mse_fn(img, val_img)))
                val_psnrs.append(np.mean(np.array(val_psnr)))

                steps.append(step)
            step += 1

    return params
