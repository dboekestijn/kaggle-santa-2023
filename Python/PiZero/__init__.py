import os
import sys

import torch

# Data parameters and functions
DATA_DIR = os.path.abspath("../PiZero/data")
MODELS_DIR = os.path.abspath("../PiZero/models")


def get_mcts_save_data_subdir(id: int, type: str, moves: int) -> str:
    return os.path.join(DATA_DIR,
                        type.replace("/", "-"),
                        "puzzle_" + str(id),
                        "moves_" + str(moves))


def get_model_save_data_subdir(type: str) -> str:
    return os.path.join(MODELS_DIR, type.replace("/", "-"))


# PiNet hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS: int = 256
NUM_RESBLOCKS: int = 40

# MCTS hyperparameters
DEFAULT_LEN_SHORTEST_PATH: int = 1_000_000
THREAD_COUNT: int = 128
MAX_SEARCH_ITERS: int = sys.maxsize
MAX_SEARCH_TIME: int = sys.maxsize
MAX_SIM_DEPTH: int = sys.maxsize
TEMPERATURE: float = 0.5
