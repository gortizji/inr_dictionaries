from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax import random


# MLP
class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu
    final_layer_sigmoid: bool = False

    @nn.compact
    def __call__(self, x):
        # x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        if self.final_layer_sigmoid:
            x = nn.sigmoid(x)

        return x


# MLP with initial mapping with fourier features
class FFN(nn.Module):
    features: Sequence[int]
    B: jnp.array
    activation: Callable = nn.relu
    final_layer_sigmoid: bool = False

    @nn.compact
    def __call__(self, x):
        x = input_mapping_fourier(x, self.B)
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        if self.final_layer_sigmoid:
            x = nn.sigmoid(x)
        return x


def input_mapping_fourier(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.0 * jnp.pi * x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class SIREN(nn.Module):
    features: Sequence[int]
    first_omega_0: float = 30
    hidden_omega_0: float = 30
    activation: Callable = jnp.sin
    input_dim: int = 1
    outermost_linear: bool = True

    @nn.compact
    def __call__(self, x):

        # first layer initialization is different than others
        feat_in = self.input_dim
        x = self.activation(
            self.first_omega_0
            * nn.Dense(
                self.features[0],
                kernel_init=my_uniform(scale=1 / feat_in),
                bias_init=my_uniform(scale=1 / feat_in),
            )(x)
        )

        # rest of the layers
        feat_in = self.features[0]
        for feat in self.features[1:-1]:
            x = self.activation(
                self.hidden_omega_0
                * nn.Dense(
                    feat,
                    kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                    bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                )(x)
            )
            feat_in = feat

        # final layer
        if self.outermost_linear:
            x = nn.Dense(
                self.features[-1],
                kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
            )(x)
        else:
            x = self.activation(
                self.hidden_omega_0
                * nn.Dense(
                    self.features[-1],
                    kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                    bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                )(x)
            )

        return x


def my_uniform(scale=1e-2, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return random.uniform(key, shape, dtype, -1, 1) * scale

    return init
