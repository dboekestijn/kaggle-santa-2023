import numpy as np

from puzzle import *


class Move:

    def __init__(self, name: str, value: np.ndarray):
        self.name = name
        self.value = value.astype(ARRAY_DTYPE)

    def get_reverse(self):
        return Move("-" + self.name, self.value.argsort())

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i: int) -> np.ushort:
        return self.value[i]  # noqa: guaranteed np.ushort

    def __str__(self):
        return "Move<" + self.name + ": " + str(self.value) + ">"

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, Move):
            return False
        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)


class MoveSet:

    def __init__(self, type: str, moves: dict[str, list[int]]):
        self.type = type
        self.moves: list[Move] = list()
        self.name_move: dict[str, Move] = dict()
        self.move_idx: dict[Move, int] = dict()
        idx = 0
        for move_name, move_tuple in moves.items():
            move = Move(move_name, np.array(move_tuple))
            self.moves.append(move)
            self.name_move[move_name] = move
            self.move_idx[move] = idx
            idx += 1

            reverse_move = move.get_reverse()
            self.moves.append(reverse_move)
            self.name_move[reverse_move.name] = reverse_move
            self.move_idx[reverse_move] = idx
            idx += 1

    def get_moves(self):
        return self.moves

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, item: int | str | Move):
        if isinstance(item, int):
            return self.moves[item]
        elif isinstance(item, str):
            return self.name_move[item]
        return self.move_idx[item]

    def __contains__(self, name: str):
        return name in self.name_move

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, MoveSet):
            return False
        return self.type == other.type and len(self) == len(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.type)
