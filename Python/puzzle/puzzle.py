import torch

from puzzle import *
from puzzle.moves import Move


class State:

    def __init__(self, value: np.ndarray):
        self.value = value.astype(ARRAY_DTYPE)
        self.hash_code: int | None = None

    def apply_move(self, move: Move):
        return State(self.value[move.value])

    def apply_moves(self, *moves: Move):
        new_value = self.value[moves[0]]
        if len(moves) > 1:
            temp_value = [0] * len(self)
            for move in moves[1:]:
                i = -1
                while (i := i + 1) < len(self):
                    temp_value[i] = new_value[move[i]]
                while (i := i - 1) >= 0:
                    new_value[i] = temp_value[i]
        return State(new_value)

    def num_matching_facelets(self, other, n_wildcards: int) -> int:
        if other is None:
            return 0
        if self is other:
            return len(self)

        n_match = sum(v == other[i] for i, v in enumerate(self.value))
        return max(n_match + n_wildcards, len(self))

    def equals_upto(self, other, n_wildcards: int) -> bool:
        if other is None:
            return False
        if self is other:
            return True
        if not isinstance(other, State):
            return False
        if len(self) != len(other):
            return False

        n_no_match = 0
        for i, v in enumerate(self.value):
            if v != other[i]:
                if (n_no_match := n_no_match + 1) > n_wildcards:
                    return False
        return True

    def to_channeled_tensor(self, puzzle, device) -> torch.Tensor:
        return (torch.Tensor(self.value == puzzle.solution_state.value)
                .reshape(1, puzzle.C, puzzle.H, puzzle.W)).to(device)

    def copy(self):
        return State(self.value.copy())

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i: int):
        return self.value[i]

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, State):
            return False
        return np.array_equal(self.value, other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self.hash_code is None:
            self.hash_code = hash(tuple(self.value))
        return self.hash_code


class Puzzle:

    @staticmethod
    def initialize_as_state(state: list[str],
                            sym_num: dict[str, int]) -> State:
        return State(np.array(tuple(sym_num[sym] for sym in state)))

    @staticmethod
    def get_tensor_dimensions(type: str) -> tuple[int, int, int]:
        if type.startswith("cube"):
            H = W = int(type[-1])
            return 6, H, W
        elif type.startswith("wreath"):
            H = 1
            W = 2 * int(type[-1]) - 2
            return 2, H, W
        else:  # type_.startswith("globe")
            H = 1 + int(type[-3])
            W = 2 * int(type[-1])
            return 1, H, W

    def __init__(self, id: int, type: str,
                 initial_state: list[str], solution_state: list[str],
                 n_wildcards: int):
        self.id = id
        self.type = type
        self.C, self.H, self.W = Puzzle.get_tensor_dimensions(type)
        self.n_wildcards = n_wildcards

        sym_num: dict[str, int] = dict()
        num = -1
        for sym in solution_state:
            num = sym_num.get(sym, num + 1)
            sym_num[sym] = num

        self.initial_state = Puzzle.initialize_as_state(initial_state, sym_num)
        self.current_state = Puzzle.initialize_as_state(initial_state, sym_num)
        self.solution_state = Puzzle.initialize_as_state(
            solution_state, sym_num)

        self.default_len_shortest_path = DEFAULT_LEN_SHORTEST_PATH

    def apply_move(self, move: Move):
        self.current_state = self.current_state.apply_move(move)

    def apply_moves(self, *move: Move):
        self.current_state = self.current_state.apply_moves(*move)

    def is_solved(self) -> bool:
        return self.current_state.equals_upto(
            self.solution_state, self.n_wildcards)

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, Puzzle):
            return False
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)
