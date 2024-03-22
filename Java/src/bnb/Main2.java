package bnb;

import data.DataLoader;
import objects.Move;
import objects.MoveSet;
import objects.Puzzle;
import objects.State;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;

public class Main2 {

    public static void main(final String[] args) throws IOException, InterruptedException, ExecutionException {
        Locale.setDefault(Locale.getDefault());

        /* --- LOAD DATA --- */
        // specify puzzle id
        final int puzzleID = 0;

        // load puzzle
        final Puzzle puzzle = DataLoader.loadPuzzle(puzzleID);
        assert puzzle != null;

        // load corresponding move set
        final MoveSet moveSet = DataLoader.loadMoveSet(puzzle.type);
        assert moveSet != null;

        /* --- SHUFFLE PUZZLE --- */
        final int nRandomMoves = 5;
        final Puzzle shuffledPuzzle = shufflePuzzle(puzzle, moveSet, nRandomMoves);

        /* --- CREATING MOVE TREE --- */
        final int maxMoves = 10; // TODO
        final int branchingFactor = moveSet.size();

        System.out.printf("Creating move tree with branching factor %d.\n", branchingFactor);
        System.out.println();
        final Tree bnbTree = new Tree(shuffledPuzzle, moveSet);

        System.out.printf("Branching and pruning (max moves set to %d) ...\n", maxMoves);
        long t0 = System.currentTimeMillis();
        final boolean foundTerminalPath = bnbTree.branchAndPrune(maxMoves);
        if (!foundTerminalPath) {
            System.out.printf("Did not find terminal path within %d moves.\n", maxMoves);
            return;
        }

        long dt = System.currentTimeMillis() - t0;
        final long nNodes = bnbTree.size();
        System.out.printf("Tree grew to %d nodes in %d seconds.\n", nNodes, dt / 1_000);
        System.out.println();

        System.out.println("Aggregating terminal paths ...");
        t0 = System.currentTimeMillis();
        final List<List<Integer>> terminalPaths = bnbTree.getMovePaths();
        dt = System.currentTimeMillis() - t0;
        System.out.printf("Aggregated %d terminal paths in %d seconds.\n", terminalPaths.size(), dt / 1_000);
        System.out.println();

        printTerminalPaths(terminalPaths);
        System.out.println();

        /* --- CREATING DIRECTED GRAPH --- */
        System.out.println("Transforming move paths into a directed graph ...");
        final DiGraph digraph = new DiGraph(shuffledPuzzle, moveSet, terminalPaths);
        System.out.printf("Graph created with %d vertices and %d arcs.\n",
                digraph.getVertices().size(), digraph.getArcs().size());
        System.out.println();

        System.out.println("Pruning graph ...");
        digraph.pruneGraph();
        System.out.printf("Pruned graph has %d vertices and %d arcs left.\n",
                digraph.getVertices().size(), digraph.getArcs().size());
        System.out.println();


        System.exit(0);
    }

    private static Puzzle shufflePuzzle(final Puzzle puzzle, final MoveSet moveSet, final int nMoves) {
        final List<Move> moves = moveSet.getMoves().stream().toList();
        final List<Move> randomMoves = new ArrayList<>();
        final Random random = new Random();
        for (int i = 0; i < nMoves; i++)
            randomMoves.add(moves.get(random.nextInt(moves.size())));
        final State currentState = puzzle.solutionState.applyMoves(randomMoves);

        final Puzzle shuffledPuzzle =
                new Puzzle(puzzle.id, puzzle.type, currentState, puzzle.solutionState, puzzle.nWildcards);
        shuffledPuzzle.setDefaultLenShortestPath(nMoves);
        return shuffledPuzzle;
    }

    private static void printTerminalPaths(final List<List<Integer>> terminalPaths) {
        int lenShortestTerminalPath = Integer.MAX_VALUE, idxShortestTerminalPath = -1, lenTerminalPath;
        for (int i = 0; i < terminalPaths.size(); ++i)
            if ((lenTerminalPath = terminalPaths.get(i).size()) < lenShortestTerminalPath) {
                lenShortestTerminalPath = lenTerminalPath;
                idxShortestTerminalPath = i;
            }

        System.out.printf("*** Shortest terminal path found of length %d ***\n", lenShortestTerminalPath);
        System.out.println(terminalPaths.get(idxShortestTerminalPath));
        System.out.println();

        System.out.println("Printing all terminal paths:");
        if (terminalPaths.size() <= 6) {
            terminalPaths.forEach(System.out::println);
            return;
        }

        final int nTerminalPaths = terminalPaths.size();
        terminalPaths.subList(0, 3).forEach(System.out::println); // Print first 3 paths (idx <=2).
        for (int i = 0; i < 3; ++i)
            System.out.println("\t.");

        // Print last 3 paths. There are >=7 terminal paths, so at least skip path 4 and start from path >=5 (idx >=4).
        terminalPaths.subList(Math.max(4, nTerminalPaths - 3), nTerminalPaths).forEach(System.out::println);
    }

}
