import csv
import math
import time
from typing import Any

import numpy as np
from torch.nn import functional as F

from PiZero import *
from PiZero.pnn import PiNet
from PiZero.simulator import Simulator
from PiZero.training import CustomDataset

from puzzle.moves import Move, MoveSet
from puzzle.puzzle import State, Puzzle


class MCNode:

    id_tracker: int = 0

    def __init__(self, parent: Any, state: State,
                 id: int = -1, level: int = -1,
                 move_child=None, child_move=None):
        if id > -1:
            self.id: int = id
        else:
            self.id: int = MCNode.id_tracker
            MCNode.id_tracker += 1

        self.parent: MCNode = parent
        self.state: State = state

        if parent is None:
            assert level <= 0, "parentless node can only be at level 0"
            self.level: int = 0
        else:
            self.level: int = parent.level + 1

        self.move_child: dict[
            Move, MCNode] = dict() if move_child is None else move_child
        self.child_move: dict[
            MCNode, Move] = dict() if child_move is None else child_move

        self.policy_probits: np.ndarray | None = None

        self.len_shortest_path: int = DEFAULT_LEN_SHORTEST_PATH
        self.terminal: bool | None = None
        self.visits: int = 1

    def add_child(self, move: Move, child):
        self.move_child[move] = child
        self.child_move[child] = move

    def create_child(self, move: Move):
        return MCNode(self, self.state.apply_move(move))

    def create_and_add_child(self, move: Move):
        child = self.create_child(move)
        self.add_child(move, child)
        return child

    def remove_child(self, child):
        move = self.child_move.pop(child)
        if move is not None:
            self.move_child.pop(move)

    def get_child(self, move: Move):
        return self.move_child.get(move, None)

    def get_children(self):
        return self.child_move.keys()

    def get_unexplored_moves(self, move_set: MoveSet):
        return [move for move in move_set.get_moves()
                if move not in self.move_child]

    def get_move_to_child(self, child):
        return self.child_move.get(child, None)

    def update_value(self, len_shortest_path: int):
        self.visits += 1
        self.len_shortest_path = min(self.len_shortest_path,
                                     len_shortest_path)

    def get_score(self) -> float:
        if self.len_shortest_path == 0:
            return math.sqrt(2.)
        return 1. / math.sqrt(self.len_shortest_path)

    def get_child_probs(self, move_set: MoveSet) -> np.ndarray:
        assert self.policy_probits is not None

        child_probs = np.zeros(len(move_set))
        for i, move in enumerate(move_set.get_moves()):
            child = self.get_child(move)
            if child is not None:
                child_probs[i] = child.get_score() + \
                    ((self.policy_probits[i] * math.sqrt(self.visits)) /
                     (1 + child.visits))

        return child_probs / child_probs.sum()

    def get_unexplored_child_probs(self, move_set: MoveSet) -> np.ndarray:
        assert self.policy_probits is not None

        unexplored_child_probs = np.array([
            self.policy_probits[i] if self.get_child(move) is None else 0.0
            for i, move in enumerate(move_set.get_moves())
        ])
        return unexplored_child_probs / unexplored_child_probs.sum()

    def is_leaf(self, move_set: MoveSet):
        return len(self.move_child) < len(move_set)

    def is_terminal(self, puzzle: Puzzle):
        if self.terminal is None:
            self.terminal = self.state.equals_upto(
                puzzle.solution_state, puzzle.n_wildcards)
        return self.terminal

    def copy(self):
        return MCNode(self.parent, self.state.copy(),
                      id=self.id, level=self.level)

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, MCNode):
            return False
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)


class MCTS:

    def __init__(self, puzzle: Puzzle, move_set: MoveSet):
        self.device = DEVICE
        self.PNN = PiNet(puzzle.C, puzzle.H, puzzle.W, len(move_set),
                         hidden_channels=HIDDEN_CHANNELS,
                         num_resblocks=NUM_RESBLOCKS).to(self.device)

        self.puzzle = puzzle
        self.move_set = move_set
        self.root = MCNode(None, puzzle.initial_state)

        self.len_shortest_path: int | None = None
        self.shortest_move_path: list[Move] | None = None

    def search(self, max_iters: int = MAX_SEARCH_ITERS,
               time_limit: int = MAX_SEARCH_TIME,
               max_sim_depth: int = MAX_SIM_DEPTH):
        self.len_shortest_path = sys.maxsize
        self.shortest_move_path = None

        simulator = Simulator(max_sim_depth,
                              self.puzzle.solution_state,
                              self.move_set)

        search_start = time.time()
        for it in range(max_iters):
            if time.time() - search_start > time_limit:
                break

            # selection
            node = self.select_leaf_node(self.root)

            # the selected node may not have its policy probits assigned yet
            if node.policy_probits is None:
                # ask the network to output a policy for this node's state
                input = node.state.to_channeled_tensor(self.puzzle,
                                                       self.device)
                policy, _ = self.PNN(input)
                node.policy_probits = F.softmax(
                    policy, dim=1).detach().cpu().numpy().squeeze()

            # expansion
            node = self.expand_node(node)

            # (multithreaded) simulation
            shortest_move_path = simulator.simulate(node.state.value)

            # backpropagation
            self.backpropagate(node, shortest_move_path)

    def select_leaf_node(self, node: MCNode) -> MCNode:
        """
        Selects a leaf node by traversing the tree along explored branches
        (moves), guided by the policy of the PNN. Give a non-leaf-node's state,
        the PNN outputs a policy over all possible moves, whether explored or
        unexplored. The density of this policy is redistributed over only
        explored moves by restricting the density over unexplored moves to
        zero.
        :param node: a starting node (the root of the MCTS tree)
        :return: a leaf node
        """

        with (torch.no_grad()):
            while not node.is_leaf(self.move_set):
                if node.policy_probits is None:
                    # ask the network to output a policy for this node's state
                    input = node.state.to_channeled_tensor(self.puzzle,
                                                           self.device)
                    policy = self.PNN.policy_forward(input)
                    node.policy_probits = F.softmax(
                        policy, dim=1).detach().cpu().numpy().squeeze()

                # select the most promising move and the corresponding child
                probs = node.get_child_probs(self.move_set)
                move = self.move_set[int(probs.argmax())]
                node = node.get_child(move)

        return node

    def expand_node(self, node: MCNode) -> MCNode:
        """
        Expands the node by executing a random unexplored move and creating the
        resulting child node. Adds the child node to this node before
        returning.
        :param node: the node to expand
        :return: a random child of the expanded node
        """

        probs = node.get_unexplored_child_probs(self.move_set)
        move = self.move_set[np.random.choice(len(probs), p=probs)]
        return node.create_and_add_child(move)

    def backpropagate(self, to_node: MCNode,
                      shortest_move_path: list[int] | None):
        if not shortest_move_path:
            node = to_node
            while node is not None:
                node.update_value(self.puzzle.default_len_shortest_path)
                node = node.parent
            return

        len_shortest_path = len(shortest_move_path)
        remaining_move_path = []
        node = to_node
        while True:
            node.update_value(len_shortest_path)
            parent_node = node.parent
            if parent_node is None:
                break

            remaining_move_path.append(
                self.move_set[parent_node.get_move_to_child(node)])
            len_shortest_path += 1
            node = parent_node

        if len_shortest_path < self.len_shortest_path:
            self.len_shortest_path = len_shortest_path
            self.shortest_move_path = \
                list(reversed(remaining_move_path)) + shortest_move_path

            improvement = ((self.len_shortest_path -
                            self.puzzle.default_len_shortest_path) /
                           (0.01 * self.puzzle.default_len_shortest_path))
            print("Found new shortest path from root to solution: "
                  "%d (%.1f%%)" % (self.len_shortest_path, improvement))

    def print_tree(self):
        self.node_number = 0
        print("number,level,visits,len_SP,score,state")
        self.print_tree_rec(self.root)

    def print_tree_rec(self, node: MCNode):
        print(self.get_node_info(node))
        self.node_number += 1

        for child in node.get_children():
            self.print_tree_rec(child)

    def get_node_info(self, node: MCNode) -> str:
        return ("%d,%d,%d,%d,%g,%s" %
                (self.node_number, node.level, node.visits,
                 node.len_shortest_path, node.get_score(), str(node.state)))

    def save_data(self, filename: str, omit_zeros: bool = False):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=CustomDataset.DELIM)
            writer.writerow(MCTS.data_header())
            for data_row in self.data_iter(self.root):
                if data_row is not None:
                    writer.writerow(data_row)

    @staticmethod
    def data_header() -> list[str]:
        return ["state", "policy_target", "value_target"]

    def data_iter(self, node: MCNode, omit_zeros: bool = False) -> \
            list[str] | None:
        yield self.get_node_data(node, omit_zeros=omit_zeros)
        for child in node.get_children():
            yield from self.data_iter(child)

    def get_node_data(self, node: MCNode, omit_zeros: bool = True) -> \
            list[str] | None:
        if node is None:
            return None

        state = node.state.value.tolist()
        policy_target = [0.] * len(self.move_set)
        all_zeros = True
        for i, move in enumerate(self.move_set.get_moves()):
            child = node.get_child(move)
            if child is not None:
                policy_target[i] = child.visits / node.visits
                all_zeros = False
        if all_zeros:
            return None

        value_target = node.get_score()
        return [str(state), str(policy_target), str(value_target)]
