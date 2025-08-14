# imports
import numpy as np
import aigs
from aigs import State, Env


# %% Setup
env: Env


# %%
def random(state: State) -> int:
    return np.random.choice(np.where(state.legal))


def human(state: State) -> int:
    raise NotImplementedError


def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -1 if maxim else 1
        for action in np.where(state.legal):
            value = minimax(env.step(state, action), not maxim)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError


def monte_carlo(state: State, maxim: bool, compute: int) -> int:
    raise NotImplementedError  # you do this


def main(cfg):
    global env
    env = aigs.make(cfg.game)
    state = env.init()
    while not state.ended:
        # print(cfg)
        # exit()
        values = []
        for action in np.where(state.legal):
            value = minimax(env.step(state, action), not state.maxim)
            values.append(value)
        # minimax(state, state.maxim)
        state: State = env.step(state, action)
