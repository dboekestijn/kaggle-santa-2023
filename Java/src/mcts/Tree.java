package mcts;

import objects.MoveSet;

import java.util.*;

public class Tree {

    public class Node {

        private static long id_tracker = 0;

        private final long id;

        private final Node parent;

        private final short[] state;

        private final Map<String, Node> moveName_child;

        private int improvements, visits;

        private int lenShortestPath;

        public Node(final Node parent, final short[] state) {
            this.id = id_tracker++;
            this.parent = parent;
            this.state = state;
            moveName_child = new HashMap<>();
            improvements = 0;
            visits = 0;
            lenShortestPath = Integer.MAX_VALUE;
        }

        public final Node generateChild(final int[] move) {
            final short[] newState = new short[state.length];
            for (int i = 0; i < state.length; i++)
                newState[i] = state[move[i]];
            return new Node(this, newState);
        }

        public final void addChild(final String moveName, final Node node) {
            moveName_child.put(moveName, node);
        }

        public final Node generateAndAddChild(final String moveName, final int[] move) {
            final Node child = generateChild(move);
            addChild(moveName, child);
            return child;
        }

        public final List<Node> getExploredChildren() {
            return moveName_child.values().stream().filter(child -> child.getValue() > 0d).toList();
        }

        public final List<String> getUnexploredMoveNames(final MoveSet moveSet) {
            return moveSet.getMoveNames()
                    .stream()
                    .filter(name -> !moveName_child.containsKey(name) || moveName_child.get(name).getValue() == 0d)
                    .toList();
        }

        public final void updateLenShortestPath(final int lenShortestPath) {
            if (lenShortestPath < this.lenShortestPath) {
                ++improvements;
                this.lenShortestPath = lenShortestPath;
            }

            ++visits;
        }

        public final double getValue() { // higher values are better
            return lenShortestPath == Integer.MAX_VALUE ?
                    0d : Math.sqrt(improvements / ((double) visits)) / lenShortestPath;
        }

        public final boolean isLeaf(final MoveSet moveSet) {
            return moveName_child.size() < moveSet.size();
        }

        public final boolean isTerminal(final short[] solutionState) {
            return Arrays.equals(state, solutionState);
        }

        @Override
        public final boolean equals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;
            return id == ((Node) o).id;
        }

        @Override
        public final int hashCode() {
            return Objects.hash(id);
        }

    }

    private final Node root;

    public Tree(final short[] initialState) {
        this.root = new Node(null, initialState);
    }

    public final Node getRoot() {
        return root;
    }

}
