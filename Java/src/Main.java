import data.DataLoader;
import mcts.MCTS;
import mcts.util.Multinomial;
import objects.Move;
import objects.MoveSet;
import objects.Puzzle;
import objects.State;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;

public class Main {

    public static void main(String... args) throws IOException, ExecutionException, InterruptedException {
        Locale.setDefault(Locale.getDefault());

        /* --- LOAD DATA --- */
        // specify puzzle id
        final int puzzleID = 30;

        // load puzzle
        final Puzzle puzzle = DataLoader.loadPuzzle(puzzleID);
        assert puzzle != null;

        // load corresponding move set
        final MoveSet moveSet = DataLoader.loadMoveSet(puzzle.type);
        assert moveSet != null;

        // load sample submission moves list
        final List<Move> movesList = DataLoader.loadSampleSubmissionMoves(puzzle.id, moveSet);
        assert movesList != null;

        // set starting path length for puzzle
        puzzle.setDefaultLenShortestPath(movesList.size() * 10);

        /* --- SHUFFLE PUZZLE --- */
//        final int nRandomMoves = 3;
//        final List<Move> moves = moveSet.getMoves().stream().toList();
//        final List<Move> randomMoves = new ArrayList<>();
//        final Random random = new Random();
//        for (int i = 0; i < nRandomMoves; i++)
//            randomMoves.add(moves.get(random.nextInt(moves.size())));
//        final State currentState = puzzle.solutionState.applyMoves(randomMoves);
//
//        final Puzzle shuffledPuzzle =
//                new Puzzle(puzzle.id, puzzle.type, currentState, puzzle.solutionState, puzzle.nWildcards);
//        shuffledPuzzle.setDefaultLenShortestPath(nRandomMoves);

        final Puzzle shuffledPuzzle = puzzle;

        /* --- DO MCTS --- */
        final MCTS mcts = new MCTS(shuffledPuzzle, moveSet);
        mcts.search(Integer.MAX_VALUE, 60L, shuffledPuzzle.getDefaultLenShortestPath());

//        final List<Move> shortestMovePath = mcts.getShortestMovePath().stream().map(moveSet::getMove).toList();
//        shuffledPuzzle.applyMoves(shortestMovePath);
//        final int nMoves = shortestMovePath.size();
//        System.out.printf("Puzzle solved in %d %s: %b\n",
//                nMoves, nMoves > 1 ? "moves" : "move", shuffledPuzzle.isSolved());
//        System.out.println(shortestMovePath);

        mcts.printTree();

        System.exit(0);
    }

}
