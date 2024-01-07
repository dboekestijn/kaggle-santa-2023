from concurrent.futures import ThreadPoolExecutor

import numpy as np
from mcts_clib import simulator

from PiZero import THREAD_COUNT
from puzzle.moves import MoveSet
from puzzle.puzzle import State


class Simulator:

    def __init__(self, max_sim_depth: int,
                 solution_state: State,
                 move_set: MoveSet):
        self.simulator = simulator.Simulator(
            THREAD_COUNT,
            max_sim_depth,
            solution_state.value,
            [move.value for move in move_set.get_moves()]
        )

    def get_thread_count(self) -> int:
        return self.simulator.getThreadCount()

    def get_max_sim_depth(self) -> int:
        return self.simulator.getMaxSimDepth()

    def get_move(self, i: int) -> list[int]:
        return self.simulator.getMove(i)

    def get_moves(self) -> list[list[int]]:
        return self.simulator.getMoves()

    def simulate(self, from_state: np.ndarray) -> list[int] | None:
        return self.simulator.getSimulationResults(from_state)
