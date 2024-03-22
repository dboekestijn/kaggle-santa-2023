import os
import sys

import numpy as np
import torch

from puzzle.puzzle import State, Puzzle

# Data parameters and functions
DATA_DIR = os.path.abspath("../PiZero/data")
MODELS_DIR = os.path.abspath("../PiZero/models")


def get_mcts_save_data_subdir(id: int, type: str, moves: int) -> str:
    return os.path.join(DATA_DIR,
                        type.replace("/", "-"),
                        "puzzle_" + str(id),
                        "moves_" + str(moves))


def get_model_save_data_subdir(puzzle: Puzzle) -> str:
    return os.path.join(MODELS_DIR,
                        puzzle.type.replace("/", "-"),
                        f"f{np.unique(puzzle.solution_state.value).size:d}")


# PiNet hyperparameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO
HIDDEN_CHANNELS: int = 64
NUM_RESBLOCKS: int = 10


def get_input_tensor(current_state: State | np.ndarray, puzzle: Puzzle,
                     device=None) -> torch.Tensor:
    # if isinstance(current_state, State):
    #     a = current_state.value[np.newaxis, :]  # (1, H): current_state here is best
    # else:
    #     a = current_state[np.newaxis, :]
    # b = puzzle.solution_state.value[:, np.newaxis]  # (H, 1): solution_state here is best
    # t = torch.from_numpy(
    #     (a == b).astype(np.float32)[:, :, np.newaxis]  # (H, H, 1)
    # )
    # return t if device is None else t.to(device)

    # if isinstance(current_state, State):
    #     a = current_state.value[np.newaxis, :]  # (1, H)
    # else:
    #     a = current_state[np.newaxis, :]
    # b = puzzle.solution_state.value[:, np.newaxis]  # (H, 1)
    # t = torch.from_numpy(
    #     (a == b).flatten().astype(np.float32)  # (H^2, )
    # )
    # return t if device is None else t.to(device)

    u = np.unique(puzzle.solution_state.value)[:, np.newaxis]  # (C, 1)
    a = current_state.value[np.newaxis, :]  # (1, H)
    t = torch.from_numpy(
        (u == a)  # (C, H)
        .astype(np.float32)
        [:, :, np.newaxis]  # (C, H, 1)
    )
    return t if device is None else t.to(device)

    # b = puzzle.solution_state.value[np.newaxis, :]  # (1, H)
    # return torch.from_numpy(
    #     np.stack((
    #         u == a,
    #         u == b
    #         # (u == a) == (u == b)
    #     ), axis=2).astype(np.float32)  # (C, H, W)
    # ).to(device)


# def get_input_tensor(current_state: State, puzzle: Puzzle,
#                      device) -> torch.Tensor:
#     input_tensor = torch.zeros((puzzle.C, puzzle.H, puzzle.W))
#     for i, n in enumerate(puzzle.sym_num.values()):
#         for j in range(puzzle.H):
#             if puzzle.solution_state[j] == n:
#                 input_tensor[i, j, 0] = 1.
#             if current_state[j] == n:
#                 input_tensor[i, j, 1] = 1.
#     return input_tensor.to(device)


# MCTS hyperparameters
DEFAULT_LEN_SHORTEST_PATH: int = 1_000_000
THREAD_COUNT: int = 128
MAX_SEARCH_ITERS: int = sys.maxsize
MAX_SEARCH_TIME: int = sys.maxsize
MAX_SIM_DEPTH: int = sys.maxsize
TEMPERATURE: float = 0.5
