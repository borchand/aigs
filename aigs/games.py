# games.py
#   aigs games
# by: Noah Syrkis

# imports
from aigs.types import State, Env
import numpy as np


# ConnectFour
class ConnectFour(Env):
    def init(self) -> State:
        raise NotImplementedError()  # You should implement this method

    def step(self, state, action) -> State:
        raise NotImplementedError()  # You should implement this method

    def play(self, state) -> State:
        raise NotImplementedError()  # You should implement this method


# TicTacToe
class TicTacToe(Env):
    def init(self) -> State:
        board = np.zeros((3, 3), dtype=np.int8)
        legal = board.flatten() == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # make your move
        board = state.board.copy()
        assert board[action // 3, action % 3] == 0
        board[action // 3, action % 3] = 1 if state.maxim else 2

        # did it win?
        mask = board.copy() == (1 if state.maxim else 2)
        winner: bool = (
            mask.all(axis=1).any()  # |
            or mask.all(axis=0).any()  # —
            or mask.trace() == 3  # \
            or np.fliplr(board).trace() == 3  # /
        )

        # is the game over?
        ended = (board != 0).all() | winner

        # return the next state
        return State(
            board=board,
            legal=board.flatten() == 0,  # empty board positions
            ended=ended,
            point=(1 if state.maxim else -1) if winner else 0,
            maxim=not state.maxim,
        )

    def play(self, state: State) -> State:
        while not state.ended:
            print(state)
            action: int = int(input(f"{'x' if state.maxim else 'o'} move: "))
            state: State = self.step(state, action)
        print(
            state,
            (("x" if state.point == 1 else "o") + " won")
            if state.point != 0
            else "draw",
        )
        return state
