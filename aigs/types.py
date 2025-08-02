from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class State:
    board: np.ndarray
    legal: np.ndarray
    point: int = 0
    maxim: bool = True
    terminated: bool = False


class Env(ABC):
    @abstractmethod
    def init(self) -> State:
        pass

    @abstractmethod
    def step(self, state, action) -> State:
        pass
