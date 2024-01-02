package mcts;

import mcts.util.Multinomial;
import mcts.util.Softmax;
import objects.MoveSet;
import objects.Puzzle;
import util.Pair;

import java.util.List;
import java.util.Map;
import java.util.Random;

public class MCTS {

    private static final int max_revisits = 3;

    private static final Random random = new Random();

    private final Puzzle puzzle;

    private final Tree tree;

    private final MoveSet moveSet;

    public MCTS(final Puzzle puzzle, final MoveSet moveSet) {
        this.puzzle = puzzle;
        this.tree = new Tree(puzzle.getInitialState());
        this.moveSet = moveSet;
    }

    public final void search(final int maxIters) {
        search(maxIters, Long.MAX_VALUE);
    }

    public final void search(final long timeLimit) {
        search(Integer.MAX_VALUE, timeLimit);
    }

    public final void search(final int maxIters, final long timeLimit) {
        search(maxIters, timeLimit, Integer.MAX_VALUE);
    }

    public final void search(final int maxIters, final long timeLimit, final int maxSimDepth) {
        final long searchStart = System.currentTimeMillis();
        Pair<Tree.Node, Map<State, Integer>> node_stateVisitCounts;
        Tree.Node node;
        Map<State, Integer>

        for (int i = 0; i < maxIters; i++) {
            if (System.currentTimeMillis() - searchStart > timeLimit)
                break;

            node_stateVisitCounts = selectNonterminalLeafNode(tree.getRoot());
            node = node_stateVisitCounts.fst;
            node = expandNode(node);
            node = simulateToTerminalNode(node, maxSimDepth);
            doBackpropagation(node);
        }
    }

    private Pair<Tree.Node, Map<List<Short>, Integer>> selectNonterminalLeafNode(Tree.Node node) {
        List<Tree.Node> exploredChildren;
        double[] probabilities;

        final double moveSetSize = moveSet.size();
        List<String> unexploredMoveNames;
        String unexploredMoveName;
        int nUnexplored;
        double explorationFactor;
        while (!node.isLeaf(moveSet)) {
            unexploredMoveNames = node.getUnexploredMoveNames(moveSet); // draw randomly
            nUnexplored = unexploredMoveNames.size();
            explorationFactor = nUnexplored / moveSetSize; // higher exploration factor if more unexplored moves
            if (random.nextDouble() < explorationFactor) {
                unexploredMoveName = unexploredMoveNames.get(random.nextInt(nUnexplored));
                node = node.generateAndAddChild(unexploredMoveName, moveSet.getMove(unexploredMoveName));
            } else {
                exploredChildren = node.getExploredChildren(); // draw based on value
                probabilities = Softmax.probabilities(
                        exploredChildren.stream().mapToDouble(Tree.Node::getValue).toArray());
                node = exploredChildren.get(Multinomial.draw(probabilities));
            }
        }

        return node;
    }

    private Tree.Node expandNode(final Tree.Node node) {
        final String randomMoveName =
                moveSet.getMoveNames().stream().skip(random.nextLong(moveSet.size())).findFirst().orElseThrow();
        return node.generateAndAddChild(randomMoveName, moveSet.getMove(randomMoveName));
    }

    private Tree.Node simulateToTerminalNode(Tree.Node node, int maxSimDepth) {

        while (maxSimDepth-- > 0 && !node.isTerminal(puzzle.getSolutionState()))
            node = expandNode(node);
        return node;
    }

    private void doBackpropagation(Tree.Node node) {

    }

}
