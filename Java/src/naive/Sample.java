package naive;

import objects.Move;
import objects.MoveSet;
import objects.Puzzle;
import objects.State;

import java.util.*;
import java.util.concurrent.*;

public class Sample {

    private class Node {

        private final Node parent;

        private final State state;

        private int lenShortestPath, distanceToSolution;

        private boolean wasChecked, isSolved;

        Node(final Node parent, final State state) {
            this.parent = parent;
            this.state = state;
            lenShortestPath = Integer.MAX_VALUE;
            distanceToSolution = -1;
            wasChecked = false;
            isSolved = false;
        }

        Node applyMove(final Move move) {
            final State newState = state.applyMove(move);
            Node node = parent;
            while (node != null) {
                if (newState.equals(node.state))
                    return null;
                node = node.parent;
            }

            return new Node(this, newState);
        }

        void backpropagate(final int lenShortestPath) {
            this.lenShortestPath = Math.min(this.lenShortestPath, lenShortestPath);
            if (parent != null)
                parent.backpropagate(lenShortestPath + 1);
        }

        boolean solvesPuzzle(final Puzzle puzzle) {
            if (wasChecked)
                return isSolved;

            wasChecked = true;
            isSolved = state.equalsUpTo(puzzle.solutionState, puzzle.nWildcards);
            return isSolved;
        }

        int distanceToSolution(final Puzzle puzzle) {
            if (distanceToSolution >= 0)
                return distanceToSolution;
            return (distanceToSolution = state.distance(puzzle.solutionState, puzzle.nWildcards));
        }

        @Override
        public final boolean equals(final Object o) {
            if (this == o)
                return true;
            if (!(o instanceof Node))
                return false;
            return state.equals(((Node) o).state);
        }

        @Override
        public final int hashCode() {
            return state.hashCode();
        }

    }

    private final ExecutorService executor;

    private final Node startingNode;

    private final Puzzle puzzle;

    private final MoveSet moveSet;

    private final Set<Node> currentSample;

    private final Map<State, Integer> state_lenShortestPath;

    public Sample(final Puzzle puzzle, final MoveSet moveSet) {
        executor = Executors.newFixedThreadPool(moveSet.size());

        this.startingNode = new Node(null, puzzle.initialState);
        this.puzzle = puzzle;
        this.moveSet = moveSet;

        currentSample = new HashSet<>(List.of(startingNode));
        state_lenShortestPath = new HashMap<>();
    }

    public final boolean expand() throws ExecutionException, InterruptedException {
        // create expanded sample by applying all moves to all nodes
        final Set<Node> expandedSample = new HashSet<>();
//        final List<Callable<Boolean>> callables = Collections.synchronizedList(new ArrayList<>());
//        moveSet.getMoves().forEach(move ->
//                callables.add(() -> {
//                    currentSample.forEach(node -> expandedSample.add(node.applyMove(move)));
//                    return true;
//                })
//        );
//        for (final Future<Boolean> fut : executor.invokeAll(callables))
//            fut.get();

        currentSample.forEach(node ->
                moveSet.getMoves().forEach(move -> {
                            final Node newNode = node.applyMove(move);
                            if (newNode != null)
                                expandedSample.add(newNode);
                        })
        );

        // copy to current sample
        currentSample.clear();
        currentSample.addAll(expandedSample);

        for (final Node node : currentSample)
            if (node.solvesPuzzle(puzzle))
                return true;
        return false;
    }

    public final void filterProbabilistically(final int filterFactor, final int maxSampleSize, final boolean keepBest) {
        assert filterFactor > 0;

        int longestDistance = 0, shortestDistance = Integer.MAX_VALUE;
        for (final Node node : currentSample) {
            longestDistance = Math.max(longestDistance, node.distanceToSolution(puzzle));
            shortestDistance = Math.min(shortestDistance, node.distanceToSolution(puzzle));
        }
        System.out.printf("Longest/shortest distance in unfiltered sample of size %d: %d/%d.\n",
                currentSample.size(), longestDistance, shortestDistance);

        // clear original list after saving into temp copy
        final List<Node> sampleList = new ArrayList<>(currentSample);
        currentSample.clear();

        // add best (optional)
        final int bestDistance = shortestDistance;
        if (keepBest)
            sampleList.forEach(node -> {
                if (node.distanceToSolution(puzzle) == bestDistance)
                    currentSample.add(node);
            });

        final int maxDistance = sampleList.stream()
                .mapToInt(node -> node.distanceToSolution(puzzle)).max().orElseThrow();

        final int n = sampleList.size();
        final double[] probabilities = new double[n];
        double sum = 0d;
        int i = -1;
        while (++i < n) { // start as unweighted values / logits
            if (sampleList.get(i).distanceToSolution(puzzle) == bestDistance)
                continue;

            probabilities[i] = maxDistance - sampleList.get(i).distanceToSolution(puzzle);
            sum += probabilities[i];
        }
        while (--i >= 0) // weight the values into probabilities
            probabilities[i] /= sum;

        final int nToKeep = Math.min(maxSampleSize, (int) Math.floor(n / ((double) filterFactor))) -
                currentSample.size();
        if (nToKeep > 0) {
            final List<Integer> indicesToKeep = mcts.util.Random.choice(probabilities, nToKeep);

            // copy only those nodes with index in indicesToKeep
            indicesToKeep.forEach(index -> currentSample.add(sampleList.get(index)));
        }

        longestDistance = 0; shortestDistance = Integer.MAX_VALUE;
        for (final Node node : currentSample) {
            longestDistance = Math.max(longestDistance, node.distanceToSolution(puzzle));
            shortestDistance = Math.min(shortestDistance, node.distanceToSolution(puzzle));
        }
        System.out.printf("Longest/shortest distance in filtered sample of size %d: %d/%d.\n",
                currentSample.size(), longestDistance, shortestDistance);
    }

    public final void filterTop(final int filterFactor, final int maxSampleSize) {
        assert filterFactor > 0;

        final int n = currentSample.size();
        final int nToFilter = n - Math.min(maxSampleSize, (int) Math.floor(n / ((double) filterFactor)));

        int longestDistance = 0, shortestDistance = Integer.MAX_VALUE;
        for (final Node node : currentSample) {
            longestDistance = Math.max(longestDistance, node.distanceToSolution(puzzle));
            shortestDistance = Math.min(shortestDistance, node.distanceToSolution(puzzle));
        }
        System.out.printf("Longest / shortest distance in unfiltered sample: %d / %d.\n",
                longestDistance, shortestDistance);

        final List<Node> nodesToKeep = currentSample.parallelStream()
                // sort from high to low distance (reverse order)
                .sorted((a, b) -> -Integer.compare(a.distanceToSolution(puzzle), b.distanceToSolution(puzzle)))
                // skip the worst nToFilter nodes (highest distance)
                .skip(nToFilter).toList();
        System.out.println("Filtered from " + n + " to " + nodesToKeep.size() + " nodes.");
        longestDistance = 0; shortestDistance = Integer.MAX_VALUE;
        for (final Node node : nodesToKeep) {
            longestDistance = Math.max(longestDistance, node.distanceToSolution(puzzle));
            shortestDistance = Math.min(shortestDistance, node.distanceToSolution(puzzle));
        }
        System.out.printf("Longest / shortest distance in filtered sample: %d / %d.\n",
                longestDistance, shortestDistance);

        // copy
        currentSample.clear();
        currentSample.addAll(nodesToKeep);
    }

    public final void filterRandomly(final int filterFactor) {
        if (filterFactor == 1)
            return;
        assert filterFactor > 0;

        // shuffle sample indices
        final int n = currentSample.size();
        final List<Integer> allIndices = new ArrayList<>();
        for (int i = 0; i < n; ++i)
            allIndices.add(i);
        Collections.shuffle(allIndices);

        // skip first nToKeep elements to get a sorted list of random indices to trim
        final int nToKeep = (int) Math.floor(n / ((double) filterFactor));
        final List<Integer> indicesToFilter = allIndices.stream()
                .skip(nToKeep).sorted().toList();
        final int nToFilter = indicesToFilter.size();

        // iterate over sample and drop those with index in indicesToTrim
        final Iterator<Node> sampleIterator = currentSample.iterator();
        int iteratorIdx = -1, filterIdx = 0;
        while (++iteratorIdx < n) {
            sampleIterator.next();
            if (iteratorIdx == indicesToFilter.get(filterIdx)) {
                sampleIterator.remove();
                if (++filterIdx == nToFilter)
                    break;
            }
        }
    }

    public final int backpropAndTrim() {
        final int currentSampleSize = currentSample.size();

        final Iterator<Node> sampleIterator = currentSample.iterator();
        Node node;
        while (sampleIterator.hasNext())
            if ((node = sampleIterator.next()).solvesPuzzle(puzzle))
                node.backpropagate(0);
            else
                sampleIterator.remove();

        final int trimmedSampleSize = currentSample.size();
        System.out.printf("Trimmed sample from %d nodes to %d nodes.\n",
                currentSampleSize, trimmedSampleSize);
        return trimmedSampleSize;
    }

    public final void freeze() {
        Node node;
        for (final Node sampleNode : currentSample) {
            node = sampleNode;
            int lenShortestPath;
            while (node != null) {
                lenShortestPath = state_lenShortestPath.getOrDefault(node.state, -1);
                if (lenShortestPath == -1 || node.lenShortestPath < lenShortestPath)
                    state_lenShortestPath.put(node.state, node.lenShortestPath);
                node = node.parent;
            }
        }
    }

    public final Map<State, Integer> getState2shortestPathLenMap() {
        return state_lenShortestPath;
    }

}
