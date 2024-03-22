import csv
import os
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

import PiZero
from PiZero.pnntransformer import PiNet
from PiZero.training import CustomDataset, PiZeroTrainer
from data import puzzle_files
from puzzle.moves import MoveSet
from puzzle.puzzle import Puzzle, State

UNIQUE_MOVES_DATA_DIR = os.path.abspath("../../Java/data")


def load_unique_move_sequences(type: str, max: int | None = None) -> \
        list[list[str]]:
    with open(UNIQUE_MOVES_DATA_DIR + "/" +
              type.replace("/", "-") +
              "/move_data.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header

        move_sequences: list[list[str]] = list()
        for line in tqdm(reader):
            move_sequences.append(line)
            if max is not None and max == len(move_sequences):
                break

            # if len(move_sequences) >= 100_000:
            #     break

        random.shuffle(move_sequences)
        return move_sequences


def get_reverse_move_name(move_name: str) -> str:
    if move_name.startswith("-"):
        return move_name[1:]
    else:
        return f"-{move_name}"


if __name__ == "__main__":
    EPOCHS = 1_000_000

    fst_puzzle_id = 0  # make sure it's the puzzle of the type we want to train
    fst_puzzle: Puzzle = puzzle_files.load_puzzle(fst_puzzle_id)
    move_set: MoveSet = puzzle_files.load_move_set(fst_puzzle.type)

    max_lines = 100_000
    puzzles = [puzzle_files.load_puzzle(id) for id in range(0, 1)]
    unique_move_sequences = load_unique_move_sequences(move_set.type, max_lines)

    max_n_per_dist = 500_000
    dist_n: dict[int, int] = dict()
    unique_states: set[State] = set()

    initial_device = torch.device("cpu")
    state_tensors: list[torch.Tensor] | list[np.ndarray] = list()
    policy_tensors: list[torch.Tensor] = list()
    value_tensors: list[torch.Tensor] = list()
    max_move_path_len: int = -1
    for i, puzzle in enumerate(puzzles):
        # if np.unique(puzzle.solution_state.value).size != 6:
        #     print(np.unique(puzzle.solution_state.value).size)
        #     continue
        print(f"Loading puzzle {i+1} of {len(puzzles)} ...")
        time.sleep(1)
        for move_sequence in tqdm(unique_move_sequences):
            dist = len(move_sequence)
            max_move_path_len = max(max_move_path_len, dist)
            if dist in dist_n and dist_n[dist] == max_n_per_dist:
                continue

            resulting_state = puzzle.solution_state.apply_moves(
                *[move_set[move_name] for move_name in move_sequence])
            state_tensors.append(torch.LongTensor(
                resulting_state.value.astype(np.int16)))
            move_idx = move_set[
                move_set[get_reverse_move_name(move_sequence[-1])]]
            policy_tensors.append(
                torch.LongTensor([move_idx]).to(initial_device))
            value_tensors.append(
                torch.FloatTensor([dist]).to(initial_device))

    custom_dataset = CustomDataset(state_tensors,
                                   policy_tensors,
                                   value_tensors)

    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    H = fst_puzzle.H
    C = fst_puzzle.C
    n_transformer_layers = 10  # TODO
    num_moves = len(move_set)
    d_model = 64
    nhead = 16
    dim_feedforward = 256  # advised: 4xd_model
    pinet = PiNet(
        C, H, n_transformer_layers, num_moves,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward
    ).to(PiZero.DEVICE)

    state_dict = torch.load("../PiZero/models/cube_2-2-2/f6/pnn_h64_r10.pth")
    pinet.load_state_dict(state_dict)

    pizero_trainer = PiZeroTrainer(pinet, fst_puzzle)
    pizero_trainer.train(EPOCHS, custom_dataset, batch_size=8_000)
