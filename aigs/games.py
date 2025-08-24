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
        # hint: use x.diagonal(i)
        raise NotImplementedError


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
