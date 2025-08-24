# games.py
#   aigs games
# by: Noah Syrkis

# imports
from aigs.types import State, Env
import numpy as np


# connect four
class ConnectFour(Env):
    def init(self) -> State:
        board = np.zeros((6, 7), dtype=int)
        legal = board[0] == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        board = state.board.copy()
        assert board[0, action] == 0, "Column already full"
        board[np.argmax(board[:, action] != 0) - 1, action] = 1 if state.maxim else -1

        # evaluate the thing
        mask = board == (1 if state.maxim else -1)
        seqs = [mask.diagonal(i) for i in range(-mask.shape[0] + 4, mask.shape[1] - 3)]  # left diags
        seqs += [mask.T.diagonal(i) for i in range(-mask.T.shape[0] + 4, mask.T.shape[1] - 3)]  # rights diags
        seqs += [mask[i] for i in range(mask.shape[0])]  # cols
        seqs += [mask[:, i] for i in range(mask.shape[1])]  # rows
        aux = lambda x: True in [x[i] and x[i + 1] and x[i + 2] and x[i + 3] for i in range(len(x) - 3)]  # noqa
        winner = True in list(map(aux, seqs))  # test for 4 consecutive pieces

        return State(
            board=board,
            legal=board[0] == 0,
            ended=winner | (board.sum() == 42),
            point=(1 if state.maxim else -1) if winner else 0,
            maxim=not state.maxim,
        )


# tic tac toe
class TicTacToe(Env):
    def init(self) -> State:
        board = np.zeros((3, 3), dtype=int)
        legal = board.flatten() == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # make your move
        board = state.board.copy()
        assert board[action // 3, action % 3] == 0, f"Invalid move: {action}"
        board[action // 3, action % 3] = 1 if state.maxim else -1

        # was it a winning move?
        mask = board == (1 if state.maxim else -1)
        winner: bool = (
            mask.all(axis=1).any()  # |
            or mask.all(axis=0).any()  # —
            or mask.trace() == 3  # \
            or np.fliplr(mask).trace() == 3  # /
        )

        # return the next state
        return State(
            board=board,
            legal=board.flatten() == 0,  # empty board positions
            ended=(board != 0).all() | winner,
            point=(1 if state.maxim else -1) if winner else 0,
            maxim=not state.maxim,
        )
