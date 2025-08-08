# imports
import numpy as np
import aigs
from aigs import State, Env


# %% Setup
env: Env


# %%
def minimax(state: State, maxim: bool) -> int:
    def aux(state: State, maxim: bool):
        if state.ended:
            return state.point
        else:
            temp: int = -1 if maxim else 1
            for action in np.where(state.legal):
                value: int = aux(env.step(state, action), not maxim)
                temp: int = (max if maxim else min)(temp, value)
            return temp

    actions = np.where(state.legal)[0]
    values = np.array([aux(env.step(state, a), maxim) for a in actions])
    return values.argmax()


def alpha_beta(state: State, maxim: bool) -> int:
    def aux(state: State, maxim: bool, alpha: int, beta: int):
        if state.ended:
            return state.point
        else:
            temp: int = -1 if maxim else 1
            for action in np.where(state.legal):
                value: int = aux(env.step(state, action), not maxim, alpha, beta)
                temp: int = (max if maxim else min)(temp, value)
                if maxim:
                    alpha: int = max(alpha, temp)
                else:
                    beta: int = min(beta, temp)
                if alpha >= beta:
                    break
            return temp

    actions = np.where(state.legal)[0]
    values = np.array([aux(env.step(state, a), maxim, -1, 1) for a in actions])
    return values.argmax()

    # return np.array(
    # [aux(env.step(state, a), maxim) for a in np.where(state.legal)[0]]
    # ).argmax()


# %% Tasks
def task_1(cfg):
    global env
    env = aigs.make(cfg.game)
    state: State = env.init()  # intialized game statess

    while not state.ended:
        action = minimax(state, state.maxim)
        state: State = env.step(state, action)


def task_2(cfg):
    global env
    env = aigs.make(cfg.game)
    state: State = env.init()  # intialized game statess

    while not state.ended:
        action = alpha_beta(state, state.maxim)
        state: State = env.step(state, action)
