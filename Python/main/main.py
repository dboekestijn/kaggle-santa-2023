import math
import os
import random

from PiZero import get_mcts_save_data_subdir
from PiZero.mcts import MCTS
from data import puzzle_files

from puzzle import DEFAULT_LEN_SHORTEST_PATH
from puzzle.moves import MoveSet, Move
from puzzle.puzzle import Puzzle


def get_mcts_savefile_path(puzzle: Puzzle, moves: int) -> str:
    subdir = get_mcts_save_data_subdir(puzzle.id, puzzle.type, moves)
    filename = "Game_{}.csv"
    i = 0
    while True:
        path = os.path.join(subdir, filename.format(i))
        if not os.path.exists(path):
            os.makedirs(subdir, exist_ok=True)
            return path
        i += 1


def run_mcts(puzzle: Puzzle, move_set: MoveSet,
             n_random_moves: int, time_limit: int):
    # shuffle puzzle randomly starting from solution state
    state = puzzle.solution_state.copy()
    for _ in range(n_random_moves):
        random_move_idx = random.randint(0, len(move_set) - 1)
        state = state.apply_move(move_set[random_move_idx])
    puzzle.initial_state = state

    # set default len shortest path
    puzzle.default_len_shortest_path = DEFAULT_LEN_SHORTEST_PATH

    # run MCTS algorithm
    mcts: MCTS = MCTS(puzzle, move_set)
    mcts.search(time_limit=time_limit,
                max_sim_depth=puzzle.default_len_shortest_path)

    save_path = get_mcts_savefile_path(puzzle, n_random_moves)
    mcts.save_data(save_path)


if __name__ == "__main__":
    n_moves = (1, 2, 3, 4, 5, 10, 15, 30)
    time_limits = (30, 30, 60, 60, 60, 300, 450, 900)

    for id in range(30):
        puzzle: Puzzle = puzzle_files.load_puzzle(id)
        move_set: MoveSet = puzzle_files.load_move_set(puzzle.type)
        # sample_submission_moves: list[Move] = \
        #     puzzle_files.load_sample_submission_moves(move_set, puzzle.id)

        for i, n_random_moves in enumerate(n_moves):
            print("===== Running MCTS for puzzle", id, "with",
                  n_random_moves, "random",
                  "move" if n_random_moves == 1 else "moves",
                  "and a time limit of", time_limits[i], "seconds",
                  "====="),

            run_mcts(puzzle, move_set,
                     n_random_moves, time_limits[i])
            print()
