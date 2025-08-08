# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import jax.numpy as np
from jax import random
from pgx import State
from jaxtyping import Array
import gymnasium as gym
from gymnasium import envs
import gym_pcgrl


# %% Setup


def main(cfg):
    print("ask")
    exit()
    env = gym.make("zelda-wide-v0")
    obs = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    # print([env for env in envs.registry])  #  if "gym_pcgrl" in env.entry_point])


# rng = random.PRNGKey(0)  # jax number generatorr


# env: Env = pgx.make("tic_tac_toe")  # game environments
# state: State = env.init(rng)  # intialized game statess
# param = random.normal(rng, (3, 3))  # param init
