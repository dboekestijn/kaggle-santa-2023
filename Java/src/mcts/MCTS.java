package mcts;

import mcts.util.Softmax;
import objects.Move;
import objects.MoveSet;
import objects.Puzzle;
import objects.State;

import java.util.*;
import java.util.concurrent.*;

public class MCTS {

//    private static final int max_revisits = 1;

    private static final double max_exploration_factor = 1d;

    private static final int thread_count = 128;

    private static final Random random = new Random();

    private final Puzzle puzzle;

    private final MoveSet moveSet;

    private final Tree tree;

    private final ExecutorService executorService;

    private int lenShortestPath;

    private List<String> shortestMovePath;

    public MCTS(final Puzzle puzzle, final MoveSet moveSet) {
        this.puzzle = puzzle;
        this.moveSet = moveSet;
        this.tree = new Tree(puzzle.initialState, puzzle.getDefaultLenShortestPath());

        executorService = Executors.newFixedThreadPool(thread_count);
    }

    public final void search(final int maxIters) throws ExecutionException, InterruptedException {
        search(maxIters, Long.MAX_VALUE);
    }

    public final void search(final long timeLimit) throws ExecutionException, InterruptedException {
        search(Integer.MAX_VALUE, timeLimit);
    }

    public final void search(final int maxIters, final long timeLimit) throws ExecutionException, InterruptedException {
        search(maxIters, timeLimit, Integer.MAX_VALUE);
    }

    public final void search(final int maxIters, final long timeLimit, final int maxSimDepth)
            throws InterruptedException, ExecutionException {
        this.lenShortestPath = Integer.MAX_VALUE;
        this.shortestMovePath = null;

        Node node;
        List<Callable<Node>> simulationTasks;
        List<Node> simulationResults;

        final long timeLimitMS = timeLimit * 1000;
        final long searchStart = System.currentTimeMillis();
        for (int i = 0, j; i < maxIters; i++) {
//            if (i > 0 && i % 1_000 == 0)
//                System.out.printf("Iter %d (time since start: %ds.): Tree has %d nodes.\n",
//                        i, (System.currentTimeMillis() - searchStart) / 1_000, tree.size());

            if (System.currentTimeMillis() - searchStart > timeLimitMS)
                break;

            // selection
            node = selectLeafNode(tree.getRoot());
            if (node.getMatchingFacelets() == -1)
                node.updateMatchingFacelets(puzzle.solutionState, puzzle.nWildcards);

            // expansion
            final Node expandedNode = expandNode(node);
            expandedNode.updateMatchingFacelets(puzzle.solutionState, puzzle.nWildcards);
            tree.incrementSize();

            // (multithreaded) simulation
            simulationTasks = new ArrayList<>();
            for (j = 0; j < thread_count; j++)
                simulationTasks.add(() -> simulate(expandedNode.copy(), maxSimDepth));

            simulationResults = new ArrayList<>();
            for (final Future<Node> outcome : executorService.invokeAll(simulationTasks))
                simulationResults.add(outcome.get());

            // backpropagation
            doBackpropagation(expandedNode, simulationResults);
        }
    }

    private Node selectLeafNode(Node node) {
        List<Node> exploredChildren;
        double[] probabilities;

        while (!node.isLeaf(moveSet)) {
            exploredChildren = node.getChildren();
            probabilities = Softmax.probabilities(
                    exploredChildren.stream().mapToDouble(Node::getScore).toArray());
            node = exploredChildren.get(mcts.util.Random.multinomial(probabilities));
        }

        return node;
    }

    private Node expandNode(final Node node) {
        final List<Move> unexploredMoves = node.getUnexploredMoves(moveSet);
        final Move randomMove = unexploredMoves.get(random.nextInt(unexploredMoves.size()));
        return node.createAndAddChild(randomMove);
    }

    private Node expandNodeWithoutRevisiting(final Node node, final Set<State> visitedStates) {
        final List<Move> unexploredMoves = node.getUnexploredMoves(moveSet);
        final List<Integer> indices = new ArrayList<>();
        int i;
        for (i = 0; i < unexploredMoves.size(); i++)
            indices.add(i);
        Collections.shuffle(indices, random);

        Move randomMove;
        Node child = null;
        for (final Integer idx : indices) {
            randomMove = unexploredMoves.get(idx);
            child = node.createChild(randomMove);
            if (!visitedStates.contains(child.getState())) {
                node.addChild(randomMove, child);
                visitedStates.add(child.getState());
                break;
            }
        }

        return child;
    }

    private Node simulate(final Node fromNode, int maxSimDepth) {
        final Set<State> visitedStates = new HashSet<>();
        visitedStates.add(fromNode.getState());

        Node node = fromNode, child;
        while (maxSimDepth-- > 0 && !node.isTerminal(puzzle.solutionState, puzzle.nWildcards)) {
            child = expandNodeWithoutRevisiting(node, visitedStates);
            if (child == null)
                return node.equals(fromNode) ? null : node;
            node = child;
        }

        return node;
    }

    private void doBackpropagation(final Node toNode, final List<Node> nodes) {
        List<String> shortestMovePath = null, shortestTerminalMovePath = null;
        int lenShortestPath = Integer.MAX_VALUE, lenShortestTerminalPath = lenShortestPath;
        List<String> shortestMovePathToNode;
        int lenShortestMovePathToNode;
        boolean isTerminal, hasTerminalPath = false;
        for (final Node node : nodes) {
            if (node == null)
                continue;

            isTerminal = node.isTerminal(puzzle.solutionState, puzzle.nWildcards);
            shortestMovePathToNode = doBackpropagation(toNode, node);
            lenShortestMovePathToNode = shortestMovePathToNode.size();
            if (isTerminal) {
                if (lenShortestMovePathToNode < lenShortestTerminalPath) {
                    lenShortestTerminalPath = lenShortestMovePathToNode;
                    shortestTerminalMovePath = shortestMovePathToNode;
                    hasTerminalPath = true;
                }
            } else {
                lenShortestMovePathToNode += node.getDefaultLenShortestPath();
                if (lenShortestMovePathToNode < lenShortestPath) {
                    lenShortestPath = lenShortestMovePathToNode;
                    shortestMovePath = shortestMovePathToNode;
                }
            }
        }

        if (hasTerminalPath) { // prefer terminal paths
            lenShortestPath = lenShortestTerminalPath;
            shortestMovePath = shortestTerminalMovePath;
        }

        if (shortestMovePath == null)
            return;

        // backpropagation to root
        Node node = toNode, parentNode;
        while (true) {
            node.updateLenShortestPath(lenShortestPath++);

            parentNode = node.getParent();
            if (parentNode == null)
                break;

            shortestMovePath.add(parentNode.getMoveToChild(node).name);
            node = parentNode;
        }

        if (!hasTerminalPath)
            return;

        --lenShortestPath;
        if (lenShortestPath < this.lenShortestPath) {
            final double percentImprovement = (puzzle.getDefaultLenShortestPath() - lenShortestPath) /
                    (0.01 * puzzle.getDefaultLenShortestPath());
            this.lenShortestPath = lenShortestPath;
            this.shortestMovePath = shortestTerminalMovePath;

            System.out.printf("Found new shortest path from root to solution: %d (%.1f%% improvement)\n",
                    this.lenShortestPath, percentImprovement);
        }
    }

    private List<String> doBackpropagation(final Node toNode, Node fromNode) {
        final List<String> shortestMovePathToNode = new ArrayList<>();
        Node parentNode;
        while (!fromNode.equals(toNode)) {
            parentNode = fromNode.getParent();
            shortestMovePathToNode.add(parentNode.getMoveToChild(fromNode).name);
            fromNode = parentNode;
        }

        return shortestMovePathToNode;
    }

    public final List<String> getShortestMovePath() {
        return shortestMovePath.reversed();
    }

    private int nodeNumber;

    public final void printTree() {
        nodeNumber = 0;
        printTreeRec(tree.getRoot());
    }

    private void printTreeRec(final Node node) {
        System.out.printf("%d,%d,%d,%g,%s%n",
                nodeNumber++, node.getLevel(), node.getVisits(), node.getScore(), node.getState().toString());
        for (final Node child : node.getChildren())
            printTreeRec(child);
    }

}
