import os

import PiZero
from PiZero.mcts import MCTS
from PiZero.pnn import PiNet
from PiZero.training import CustomDataset, PiZeroTrainer
from data import puzzle_files
from puzzle.moves import MoveSet
from puzzle.puzzle import Puzzle


def get_file_paths(puzzle_type: str) -> list[str]:
    filepaths: list[str] = []
    for root, dirs, files in os.walk(PiZero.DATA_DIR):
        if len(files) == 0:
            continue

        for file in files:
            filepaths.append(str(os.path.join(root, *dirs, file)))
    return filepaths


if __name__ == "__main__":
    EPOCHS = 100

    puzzle_id = 0  # make sure it's the puzzle of the type we want to train
    puzzle: Puzzle = puzzle_files.load_puzzle(puzzle_id)
    move_set: MoveSet = puzzle_files.load_move_set(puzzle.type)
    mcts: MCTS = MCTS(puzzle, move_set)

    device = PiZero.DEVICE
    dataset: CustomDataset = CustomDataset(
        puzzle, device, *get_file_paths(puzzle.type))
    trainer: PiZeroTrainer = PiZeroTrainer(mcts.PNN, puzzle.type)
    trainer.train(EPOCHS, dataset)

