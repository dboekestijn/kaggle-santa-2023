import json
import os

from data import *
from puzzle.moves import MoveSet, Move
from puzzle.puzzle import Puzzle


class PuzzlesFile:

    PATH = os.path.join(DATA_DIR, "puzzles.csv")
    CSV_DELIM = ","
    ID_COL, TYPE_COL, SOL_COL, INIT_COL, WILDCARDS_COL = 0, 1, 2, 3, 4
    ARRAY_SEP = ";"

    @staticmethod
    def parse_values(values: list[str]) -> \
            tuple[int, str, list[str], list[str], int]:
        return (
            int(values[PuzzlesFile.ID_COL]),
            values[PuzzlesFile.TYPE_COL],
            values[PuzzlesFile.INIT_COL].split(PuzzlesFile.ARRAY_SEP),  # initial state first!
            values[PuzzlesFile.SOL_COL].split(PuzzlesFile.ARRAY_SEP),
            int(values[PuzzlesFile.WILDCARDS_COL])
        )


class PuzzleInfoFile:

    PATH = os.path.join(DATA_DIR, "puzzle_info.csv")
    CSV_DELIM = ","
    TYPE_COL, MOVES_COL = 0, 1

    @staticmethod
    def parse_values(values: list[str]) -> tuple[str, dict[str, list[int]]]:
        type_ = values[PuzzleInfoFile.TYPE_COL]
        moves = json.loads(
            values[PuzzleInfoFile.MOVES_COL].replace("'", '"'), )
        return type_, moves


class SampleSubmissionFile:

    PATH = os.path.join(SUBMISSIONS_DIR, "sample_submission.csv")
    CSV_DELIM = ","
    ID_COL, MOVES_COL = 0, 1
    MOVES_SEP = "."

    @staticmethod
    def parse_values(values: list[str]) -> tuple[int, list[str]]:
        return (
            int(values[SampleSubmissionFile.ID_COL]),
            values[SampleSubmissionFile.MOVES_COL]
            .split(SampleSubmissionFile.MOVES_SEP)
        )


def load_puzzle(id: int | str) -> Puzzle | None:
    if isinstance(id, int):
        id = str(id)

    with open(PuzzlesFile.PATH, "r") as f:
        reader = csv.reader(f, delimiter=PuzzlesFile.CSV_DELIM)
        next(reader)  # skip header
        for line in reader:
            if line[PuzzlesFile.ID_COL] == id:
                return Puzzle(*PuzzlesFile.parse_values(line))
    return None  # no puzzle with specified id found


def load_puzzles() -> list[Puzzle]:
    with open(PuzzlesFile.PATH, "r") as f:
        reader = csv.reader(f, delimiter=PuzzlesFile.CSV_DELIM)
        next(reader)  # skip header
        return [Puzzle(*PuzzlesFile.parse_values(line))
                for line in reader]


def load_move_set(type: str) -> MoveSet | None:
    with open(PuzzleInfoFile.PATH, "r") as f:
        reader = csv.reader(f, delimiter=PuzzleInfoFile.CSV_DELIM)
        next(reader)  # skip header
        for line in reader:
            if line[PuzzleInfoFile.TYPE_COL] == type:
                return MoveSet(*PuzzleInfoFile.parse_values(line))
    return None  # no move set for specified type found


def load_move_sets() -> list[MoveSet]:
    with open(PuzzleInfoFile.PATH, "r") as f:
        reader = csv.reader(f, delimiter=PuzzleInfoFile.CSV_DELIM)
        next(reader)  # skip header
        return [MoveSet(*PuzzleInfoFile.parse_values(line))
                for line in reader]


def load_sample_submission_moves(move_set: MoveSet, id: str | int = -1) -> \
        list[Move] | dict[int, list[Move]] | None:
    load_all_ids = False
    if isinstance(id, int):
        if id == -1:
            load_all_ids = True
        else:
            id = str(id)

    def get_moves(names: list[str]) -> list[Move]:
        return [move_set[move_name] for move_name in names]

    with open(SampleSubmissionFile.PATH, "r") as f:
        reader = csv.reader(f, delimiter=SampleSubmissionFile.CSV_DELIM)
        next(reader)  # skip header
        if load_all_ids:
            id_moves: dict[int, list[Move]] = dict()
            for line in reader:
                id, move_names = SampleSubmissionFile.parse_values(line)
                id_moves[id] = get_moves(move_names)
        else:
            for line in reader:
                if line[SampleSubmissionFile.ID_COL] == id:
                    _, move_names = SampleSubmissionFile.parse_values(line)
                    return get_moves(move_names)

    if load_all_ids:
        return id_moves
    return None  # no moves for puzzle with specified id found

