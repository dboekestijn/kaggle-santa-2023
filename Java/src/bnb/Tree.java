package bnb;

import objects.Move;
import objects.MoveSet;
import objects.Puzzle;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class Tree {

    private static final int core_count = 16;

    private final ExecutorService executor;

    private final Node root;

    private final short[] solutionState;

    private final MoveSet moveSet;

    private long sz;

    public Tree(final Puzzle puzzle, final MoveSet moveSet) {
        executor = Executors.newFixedThreadPool(core_count);

        this.root = new Node(null, puzzle.initialState.getValue(), moveSet.size());
        solutionState = puzzle.solutionState.getValue();
        this.moveSet = moveSet;
        sz = -1;
    }

    public final boolean branchAndPrune(final int maxMoves) throws InterruptedException, ExecutionException {
        if (root.isTerminal(solutionState))
            return true;
        if (maxMoves == 0)
            return false; // as root is not terminal

        return threadedBranchAndPrune(root, maxMoves);
    }

    private boolean threadedBranchAndPrune(final Node node, final int maxMoves) throws
            InterruptedException, ExecutionException {
        final List<Callable<Boolean>> bnpTasks = new ArrayList<>(moveSet.size());
        int childIdx = -1;
        for (final Move move : moveSet.getMoves()) {
            final int fChildIdx = ++childIdx;
            bnpTasks.add(() -> branchAndPrune(node.addChild(move.value, fChildIdx), 1, maxMoves));
        }

        boolean isTerminal = false, childIsTerminal;
        childIdx = -1;
        for (final Future<Boolean> result : executor.invokeAll(bnpTasks)) {
            ++childIdx;
            childIsTerminal = result.get();
            if (!childIsTerminal)
                node.pruneChild(childIdx);

            isTerminal = isTerminal || childIsTerminal;
        }

        return isTerminal;
    }

    private boolean branchAndPrune(final Node node, final int moveNum, final int maxMoves) {
        if (moveNum == maxMoves)
            return node.isTerminal(solutionState);

        if (node.isTerminal(solutionState)) {
            for (int childIdx = 0; childIdx < moveSet.size(); ++childIdx)
                node.pruneChild(childIdx);
            return true;
        }

        boolean isTerminal = false;
        int childIdx = -1;
        for (final Move move : moveSet.getMoves()) {
            ++childIdx;
            if (branchAndPrune(node.addChild(move.value, childIdx), moveNum + 1, maxMoves))
                isTerminal = true;
            else
                node.pruneChild(childIdx);
        }

        return isTerminal;
    }

    public final long size() {
        if (sz == -1)
            sz = size(root);
        return sz;
    }

    private long size(final Node node) {
        long sz = 1;
        for (final Node child : node.getChildren()) {
            if (child == null)
                continue;
            sz += size(child);
        }
        return sz;
    }

    public final List<List<Integer>> getMovePaths() {
        final List<List<Integer>> movePaths = new ArrayList<>();
        getMovePaths(root, -1, movePaths, new ArrayList<>());
        return movePaths;
    }

    private void getMovePaths(final Node node, final int nodeIdx,
                              final List<List<Integer>> movePaths,
                              final List<Integer> currentMovePath) {
        if (nodeIdx >= 0)
            currentMovePath.add(nodeIdx);
        if (node.isLeaf()) {
            movePaths.add(currentMovePath);
            return;
        }

        int childIdx = -1;
        for (final Node child : node.getChildren()) {
            ++childIdx;
            if (child == null)
                continue;

            getMovePaths(child, childIdx, movePaths, new ArrayList<>(currentMovePath));
        }
    }

}
