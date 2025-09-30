# imports
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field
import sys


# %% Setup
env: Env

def connect_three_test(v):
    if len(v) < 3:
        return False

    for i in range(len(v) - 2):
        if v[i] and v[i + 1] and v[i + 2]:
            return True

    return False

def connect_two_test(v):
    if len(v) < 2:
        return False

    for i in range(len(v) - 1):
        if v[i] and v[i + 1]:
            return True

    return False

def heuristic_value(board: np.ndarray) -> float:
    # mask for minim
    mask = board == -1

    rows = [row for row in mask]
    cols = [col for col in mask.T]

    row_diags = [mask.diagonal(i) for i in range(-6, 7)]
    col_diags = [np.fliplr(mask).diagonal(i) for i in range(-7, 6)]

    any_connect_three = False
    for v in rows + cols + row_diags + col_diags:
        if connect_three_test(v):
            any_connect_three = True
            break

    if any_connect_three:
        return -0.5

    any_connect_two = False
    for v in rows + cols + row_diags + col_diags:
        if connect_two_test(v):
            any_connect_two = True
            break;

    if any_connect_two:
        return -0.2

    # mask for maxim
    mask = board == 1

    rows = [row for row in mask]
    cols = [col for col in mask.T]

    row_diags = [mask.diagonal(i) for i in range(-6, 7)]
    col_diags = [np.fliplr(mask).diagonal(i) for i in range(-7, 6)]

    any_connect_three = False
    for v in rows + cols + row_diags + col_diags:
        if connect_three_test(v):
            any_connect_three = True
            break
    if any_connect_three:
        return 0.5

    any_connect_two = False
    for v in rows + cols + row_diags + col_diags:
        if connect_two_test(v):
            any_connect_two = True
            break
    if any_connect_two:
        return 0.2

    return 0


# %%
def minimax(state: State, maxim: bool, depth: int = 5) -> float:
    if state.ended or depth > 8:
        return state.point if state.ended else heuristic_value(state.board)
    else:
        temp: float = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim, depth + 1)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, depth: int, maxim: bool, alpha: float, beta: float) -> float:
    if state.ended or depth == 0:
        return state.point if state.ended else heuristic_value(state.board)
    actions  = np.where(state.legal)[0]
    if maxim:
        for action in actions:  # for all legal actions
            value = alpha_beta(env.step(state, action), depth-1, not maxim, alpha, beta)

            alpha = max(value, alpha)

            if beta <= alpha:
                break
        return alpha
    else:
        for action in actions:  # for all legal actions
            value = alpha_beta(env.step(state, action), depth-1, maxim, alpha, beta)

            beta = min(beta, value)
            if beta <= alpha:
                break
        return beta



@dataclass
class Node:
    state: State  # Add more fields
    parent: 'Node'
    children: 'list[Node]' = field(default_factory=list)
    value: float = 0.0
    visits: int = 0
    seen: list = field(default_factory=list)
    action: int = 0


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:

    n = Node(state, None)

    for _ in range(10000):

        n = tree_policy(n, cfg)

        delta = default_policy(n.state)

        backup(n, delta)

    best_child_node = best_child(n, cfg.c)

    if best_child_node.action is None:
        print("No action found")
        sys.exit(1)

    return best_child_node.action



def tree_policy(node: Node, cfg) -> Node:
    while not node.state.ended:
        e_node = expand(node)

        if not e_node is node:
            node = e_node
        else:
            node = best_child(node, cfg.c)
    return node


def expand(v: Node) -> Node:
    actions = np.where(v.state.legal)[0]

    if len(actions) == 0:
        return v

    action = np.random.choice(actions)

    while action in v.seen:
        action = np.random.choice(actions)

    v.seen.append(action)

    state = env.step(v.state, action)
    child = Node(state, v)
    child.action = action

    v.children.append(child)

    return child

def best_child(root: Node, c) -> Node:
    best_n = root
    best_v = None

    for child in root.children:
        v = child.value / child.visits + c * np.sqrt(2 * np.log(root.visits) / child.visits)
        if best_v is None or v > best_v:
            best_v = v
            best_n = child

    return best_n

def default_policy(state: State) -> float:
    while not state.ended:
        actions = np.where(state.legal)[0]
        action = np.random.choice(actions)
        state = env.step(state, action)
    return state.point + heuristic_value(state.board)


def backup(node, delta) -> None:
    while not node is None:
        node.visits += 1
        if node.state.maxim:
            node.value += delta
        else:
            node.value -= delta
        node = node.parent

# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = np.where(state.legal)[0]  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()
                print(f"random plays {a}")

            case "human":
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                print(f"minimax plays {a}")
            case "alpha_beta":
                values = [alpha_beta(env.step(state, a), 5, not state.maxim, -1000, 1000) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                print(f"alpha_beta plays {a}")

            case "monte_carlo":
                a = monte_carlo(state, cfg)
                print(f"monte_carlo plays {a}")

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)
        print(state)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
