package bnb;

import java.util.Arrays;

public class Node {

    private final Node parent;

    private final short[] state;

    private final int nMoves;

    private final Node[] children;

    private int numChildren;

    private int isTerminal;

    public Node(final Node parent, final short[] state, final int nMoves) {
        this.parent = parent;
        this.state = state;
        this.nMoves = nMoves;
        children = new Node[nMoves];
        numChildren = -1;
        isTerminal = 0;
    }

    public final Node getParent() {
        return parent;
    }

    public final short[] getState() {
        return state;
    }

    public final Node addChild(final short[] move, final int childIdx) {
        final short[] newState = new short[state.length];
        for (int i = 0; i < state.length; ++i)
            newState[i] = state[move[i]];

        numChildren = numChildren == -1 ? 1 : numChildren + 1;
        return children[childIdx] = new Node(this, newState, nMoves);
    }

    public final Node getChild(final int childIdx) {
        return children[childIdx];
    }

    public final Node[] getChildren() {
        return children;
    }

    public final void pruneChild(final int childIdx) {
        children[childIdx] = null;
        --numChildren;
    }

    public final boolean isTerminal(final short[] solutionState) {
        if (isTerminal == 0) // undefined
            isTerminal = Arrays.equals(state, solutionState) ? 1 : -1;
        return isTerminal == 1;
    }

    public final boolean isLeaf() {
        return numChildren <= 0;
    }

}
