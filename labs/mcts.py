# imports
import numpy as np
import aigs
from aigs import State, Env


# %% Setup
env: Env


# %%
def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -1 if maxim else 1
        for action in np.where(state.legal):
            value: int = minimax(env.step(state, action), not maxim)
            temp: int = (max if maxim else min)(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError


def monte_carlo(state: State, maxim: bool, compute: int) -> int:
    raise NotImplementedError  # you do this


def main(ctx):
    global env
    env = aigs.make(ctx.config.mcts.game)
    state = env.init()
    while not state.ended:
        action = np.array(
            [
                minimax(env.step(state, action), not state.maxim)
                for action in np.where(state.legal)
            ]
        ).argmax()
        # minimax(state, state.maxim)
        state: State = env.step(state, action)


# %% Tasks
def task_1(ctx):
    global env
    env = aigs.make(ctx.config.mcts.game)
    state: State = env.init()  # intialized game statess

    while not state.ended:
        action = minimax(state, state.maxim)
        state: State = env.step(state, action)


def task_2(ctx):
    global env
    env = aigs.make(ctx.config.mcts.game)
    state: State = env.init()  # intialized game statess

    # while not state.ended:
    # action = alpha_beta(state, state.maxim)
    # state: State = env.step(state, action)
