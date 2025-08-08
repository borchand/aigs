# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import jax.numpy as np
from jax import random
from pgx import State
from jaxtyping import Array
from gym import envs
import gym_pcgrl


# %% Setup
# rng = random.PRNGKey(0)  # jax number generatorr
# env: Env = pgx.make("tic_tac_toe")  # game environments
# state: State = env.init(rng)  # intialized game statess
# param = random.normal(rng, (3, 3))  # param init
