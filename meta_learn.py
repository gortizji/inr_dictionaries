import pickle

import haiku as hk
import jax.numpy as jnp
from jax import random

from models.models_haiku import SIREN
from train.meta_learn import meta_train
from utils.meta_learn import CELEBA_BUILDER, DEFAULT_GRID

BATCH_SIZE = 3
NUM_VAL_EXAMPLES = 5
INNER_LR = 0.01
OUTER_LR = 1e-5
INNER_STEPS = 2
META_METHOD = "MAML"
MAX_ITERS = 5000

model = hk.without_apply_rng(
    hk.transform(lambda x: SIREN(w0=30, width=256, hidden_w0=30, depth=5)(x))
)
init_params = model.init(random.PRNGKey(0), jnp.ones((1, 2)))

ds_train = CELEBA_BUILDER.as_dataset(
    split="train", as_supervised=False, shuffle_files=True, batch_size=BATCH_SIZE
)
ds_train = ds_train.take(-1)

ds_val = CELEBA_BUILDER.as_dataset(
    split="validation", as_supervised=False, shuffle_files=False, batch_size=1
)
ds_val = ds_val.take(NUM_VAL_EXAMPLES)

meta_params = meta_train(
    model,
    init_params,
    DEFAULT_GRID,
    ds_train,
    ds_val,
    batch_size=BATCH_SIZE,
    outer_lr=OUTER_LR,
    inner_lr=INNER_LR,
    inner_steps=INNER_STEPS,
    meta_method=META_METHOD,
    max_iters=MAX_ITERS,
)

with open("maml_celebA_5000.pickle", "wb") as handle:
    pickle.dump(meta_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
