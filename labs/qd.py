# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import jax.numpy as np
from jax import random
import pgx
from pgx import State, Env
from jaxtyping import Array


# %% Setup
rng = random.PRNGKey(0)  # jax number generatorr
env: Env = pgx.make("tic_tac_toe")  # game environments
state: State = env.init(rng)  # intialized game statess
param = random.normal(rng, (3, 3))  # param init


# %%
def action_fn(state: State, rng: Array) -> Array:
    action = random.choice(rng, np.arange(9), p=state.legal_action_mask)
    # model(state.observation, param)
    return action


def model(obs, param):
    # aigs.conv(obs, param)
    print(param)


# # %%
# while not (state.terminated | state.truncated):
#     action = action_fn(state, rng)
#     state: State = env.step(state, action)
