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
        temp: int = -10 if maxim else 10
        for a in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, a), not state.maxim)
            temp = max(temp, value) if maxim else min(temp, value)
            print(np.where(state.legal)[0])
            print("action", a, "value", value, "temp", temp, state.player)
            print(state)
            print()
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError  # you do this


# Intuitive but maybe a bit difficult in terms of code
def monte_carlo(state: State, maxim: bool, compute: int) -> int:
    raise NotImplementedError  # you do this


# Main function
def main(cfg):
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        action = action_fn(cfg, env, state)
        state = env.step(state, action)

    print(f"{['.', 'o', 'x'][state.point]} won", state, sep="\n")


# %% calls and evaluates the different kinds of action functions
def action_fn(cfg, env: Env, s: State) -> int:
    actions = np.where(s.legal)[0]  # the actions to choose from

    match getattr(cfg, s.player):
        case "random":
            return np.random.choice(actions).item()

        case "human":
            print(s, end="\n\n")
            return int(input("Place your piece: "))

        case "minimax":
            values = [minimax(env.step(s, a), not s.maxim) for a in actions]
            return np.argmax(values).item() if s.maxim else np.argmin(values).item()

        case "alpha_beta":
            values = [alpha_beta(env.step(s, a), not s.maxim, -1, 1) for a in actions]
            return np.argmax(values).item() if s.maxim else np.argmin(values).item()

        case "monte_carlo":
            raise NotImplementedError

        case _:
            raise ValueError(f"Unknown player {s.player}")
