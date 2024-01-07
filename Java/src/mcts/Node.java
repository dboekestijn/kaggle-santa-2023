package mcts;

import objects.Move;
import objects.MoveSet;
import objects.State;

import java.util.*;

public class Node {

    private static double computeScore(final int lenShortestPath, final int visits) {
        return (lenShortestPath == 0 ? 1d : 1d / Math.sqrt(lenShortestPath)) / Math.sqrt(visits);
    }

    private static long id_tracker = 0;

    private final long id;

    private final int level;

    private final Node parent;

    private final State state;

    private final Map<Move, Node> move_child;

    private final Map<Node, Move> child_move;

    private final int defaultLenShortestPath;

    private int nMatchingFacelets, lenShortestPath;

    private int visits;

    private double score;

    public Node(final Node parent, final State state) {
        this(parent, parent.level + 1, state, parent.defaultLenShortestPath);
    }

    public Node(final Node parent, final int level, final State state, final int defaultLenShortestPath) {
        this.id = id_tracker++;
        this.level = level;
        this.parent = parent;
        this.state = state;
        move_child = new HashMap<>();
        child_move = new HashMap<>();

        nMatchingFacelets = -1;

        this.defaultLenShortestPath = defaultLenShortestPath;
        lenShortestPath = Integer.MAX_VALUE; // TODO

        visits = 1;
        score = computeScore(lenShortestPath, visits);
    }

    private Node(final long id, final int level, final Node parent, final State state,
                 final int nMatchingFacelets, final int defaultLenShortestPath) {
        this.id = id;
        this.level = level;
        this.parent = parent;
        this.state = state;
        move_child = new HashMap<>();
        child_move = new HashMap<>();
        this.nMatchingFacelets = nMatchingFacelets;
        this.defaultLenShortestPath = defaultLenShortestPath;
    }

    public final long getID() {
        return id;
    }

    public final int getLevel() {
        return level;
    }

    public final Node getParent() {
        return parent;
    }

    public final State getState() {
        return state;
    }

    public final void addChild(final Move move, final Node child) {
        move_child.put(move, child);
        child_move.put(child, move);
    }

    public final Node createChild(final Move move) {
        return new Node(this, state.applyMove(move));
    }

    public final Node createAndAddChild(final Move move) {
        final Node child = createChild(move);
        addChild(move, child);
        return child;
    }

    public final void removeChildren() {
        getChildren().forEach(this::removeChild);
    }

    public final void removeChild(final Node child) {
        move_child.remove(child_move.remove(child));
    }

    public final List<Node> getChildren() {
        return move_child.values().stream().toList();
    }

    public final List<Move> getUnexploredMoves(final MoveSet moveSet) {
        return moveSet.getMoves().stream().filter(move -> !move_child.containsKey(move)).toList();
    }

    public final Move getMoveToChild(final Node child) {
        return child_move.get(child);
    }

    public final int getMatchingFacelets() {
        return nMatchingFacelets;
    }

    public final int getDefaultLenShortestPath() {
        return defaultLenShortestPath;
    }

    public final int getLenShortestPath() {
        return lenShortestPath;
    }

    public final void updateMatchingFacelets(final State solutionState, final int nWildcards) {
        nMatchingFacelets = solutionState.nMatchingFacelets(state, nWildcards);
    }

    public final void updateLenShortestPath(final int lenShortestPath) {
        if (lenShortestPath < this.lenShortestPath)
            this.lenShortestPath = lenShortestPath;
        ++visits;
        score = computeScore(this.lenShortestPath, visits);
    }

    public final double getScore() { // higher values are better
        return score;
    }

    public final int getVisits() {
        return visits;
    }

    public final boolean isLeaf(final MoveSet moveSet) {
        return move_child.size() < moveSet.size();
    }

    public final boolean isTerminal(final State solutionState) {
        return isTerminal(solutionState, 0);
    }

    public final boolean isTerminal(final State solutionState, final int nWildcards) {
        return state.equalsUpTo(solutionState, nWildcards);
    }

    public final Node copy() {
        return new Node(id, level, parent, state, defaultLenShortestPath, nMatchingFacelets);
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
