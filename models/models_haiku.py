import haiku as hk
import jax
import jax.numpy as jnp


class SIRENLayer(hk.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.out_f = out_f
        self.b = 1 / in_f if self.is_first else jnp.sqrt(6 / in_f) / w0

    def __call__(self, x):
        x = hk.Linear(
            output_size=self.out_f,
            w_init=hk.initializers.RandomUniform(-self.b, self.b),
        )(x)
        return x + 0.5 if self.is_last else self.w0 * x


class SIREN(hk.Module):
    def __init__(self, w0, width, hidden_w0, depth):
        super().__init__()
        self.w0 = w0  # to change the omega_0 of SIREN !!!!
        self.width = width
        self.depth = depth
        self.hidden_w0 = hidden_w0

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = SIRENLayer(x.shape[-1], self.width, is_first=True, w0=self.w0)(x)
        x = jnp.sin(x)

        for _ in range(self.depth - 2):
            x = SIRENLayer(x.shape[-1], self.width, w0=self.hidden_w0)(x)
            x = jnp.sin(x)

        out = SIRENLayer(x.shape[-1], 1, w0=self.hidden_w0, is_last=True)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


class MLP(hk.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.width = width
        self.depth = depth

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = hk.Linear(self.width)(x)
        x = jax.nn.relu(x)

        for _ in range(self.depth - 2):
            x = hk.Linear(self.width)(x)
            x = jax.nn.relu(x)

        out = hk.Linear(1)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


class FFN(hk.Module):
    def __init__(self, sigma, width, depth):
        super().__init__()
        self.sigma = sigma
        self.width = width
        self.depth = depth
        self.B = self.sigma * jax.random.normal(jax.random.PRNGKey(7), (width, 2))

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = input_mapping_fourier(x, self.B)

        for _ in range(self.depth - 2):
            x = hk.Linear(self.width)(x)
            x = jax.nn.relu(x)

        out = hk.Linear(1)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


def input_mapping_fourier(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.0 * jnp.pi * x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
