# imports
from __future__ import annotations
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field


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
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError  # you do this


@dataclass
class Node:
    state: State
    N: int = 0
    Q: int = 0
    parent: Node | None = None
    action: int | None = None
    expanded: bool = False
    children: list[Node] = field(default_factory=list)


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    raise NotImplementedError  # you do this


def uct_search(state: State, cfg):
    root = Node(state=state)
    for i in range(cfg.compute):
        node = tree_policy(root, cfg)
        delta = default_policy(node.state)
        backup(node, delta)
    return best_child(root, 0).action  # exploit time


def tree_policy(node: Node, cfg) -> Node:
    while not node.state.ended:
        if not node.expanded:
            return expand(node)
        else:
            node = best_child(node, 1 / np.sqrt(2))  # explore time
    return node


def expand(v: Node) -> Node:
    a: int = np.random.choice(np.where(v.state.legal)[0])
    v_prime = Node(state=env.step(v.state, a), action=a, parent=v)
    v.children.append(v_prime)
    return v_prime


def best_child(root: Node, c) -> Node:
    f = lambda node: node.Q / node.N + c * (2 * np.log(root.N) / node.N) ** 0.5  # noqa
    return root.children[np.array([f(node) for node in root.children]).argmax()]


def default_policy(state: State) -> int:
    while not state.ended:
        a = np.random.choice(np.where(state.legal)[0])
        state = env.step(state, a)
    return state.point


def backup(node, delta) -> None:
    while node is not None:
        node.N += 1
        node.Q += delta
        delta: int = -delta  # flip player
        node: Node | None = node.parent


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    s = env.init()

    while not s.ended:
        actions = np.where(s.legal)[0]  # the actions to choose from

        match getattr(cfg, s.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(s, end="\n\n")
                a = int(input("Place your piece: "))

            case "minimax":
                values = [minimax(env.step(s, a), not s.maxim) for a in actions]
                a = actions[np.argmax(values) if s.maxim else np.argmin(values)]

            case "alpha_beta":
                values = [
                    alpha_beta(env.step(s, a), not s.maxim, -1, 1) for a in actions
                ]
                a = actions[np.argmax(values) if s.maxim else np.argmin(values)]

            case "monte_carlo":
                raise NotImplementedError

            case _:
                raise ValueError(f"Unknown player {s.player}")

        s = env.step(s, a)

    print(f"{['nobody', 'o', 'x'][s.point]} won", s, sep="\n")
