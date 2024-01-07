package mcts;

import objects.Move;
import objects.State;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Tree {

    private final Node root;

    private int size;

    public Tree(final State initialState, final int defaultLenShortestPath) {
        this.root = new Node(null, 0, initialState, defaultLenShortestPath);
        size = 1;
    }

    public final Node getRoot() {
        return root;
    }

    public final Node createAndAddChild(final Node node, final Move move) {
        ++size;
        return node.createAndAddChild(move);
    }

    public final void incrementSize() {
        ++size;
    }

    public final int size() {
        return size;
    }

}
