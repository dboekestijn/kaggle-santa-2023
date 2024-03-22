package naive;

import data.DataLoader;
import objects.CompressedMove;
import objects.Move;
import objects.MoveSet;
import objects.Puzzle;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;

public class Main4 {

    public static void main(final String[] args) throws IOException, InterruptedException, ExecutionException {
//        final Map<Integer, Integer> idx_count = new HashMap<>();
//        List<Integer> chosenIndices;
//        for (int j = 0; j < 1_000_000; ++j) {
//            chosenIndices = mcts.util.Random.choice(new double[] {0.3, 0.1, 0.4, 0.2}, 1);
//            chosenIndices.forEach(idx -> idx_count.put(idx, idx_count.getOrDefault(idx, 0) + 1));
//        }
//
//        final double totalCount = idx_count.values().stream().mapToInt(Integer::intValue).sum();
//        idx_count.forEach((idx, count) -> System.out.println(idx + ": " + (count / totalCount)));
//        System.exit(0);

        /* --- LOAD DATA --- */
        // set puzzle ID
        final int puzzleID = 283;

        // load puzzle
        final Puzzle puzzle = DataLoader.loadPuzzle(puzzleID);
        assert puzzle != null;
        System.out.printf("Loaded puzzle of type '%s'.\n", puzzle.type);

        // load corresponding move set
        final MoveSet moveSet = DataLoader.loadMoveSet(puzzle.type);
        assert moveSet != null;

        final Sample sample = new Sample(puzzle, moveSet);
        final int filterFactor = 10, maxSampleSize = 10_000;
        final boolean keepBest = true;

        int i = 0;
        while (true) {
            System.out.println("Iter " + ++i + ":");
            System.out.println("Expanding ...");
            if (sample.expand())
                break;

            System.out.println("Filtering ...");
            sample.filterProbabilistically(filterFactor, maxSampleSize, keepBest);
            System.out.println();

//            Thread.sleep(2_000L);
        }

        System.out.println("Found at least one node that solves the puzzle.");
        System.out.println("Backpropagating results ...");
        sample.backpropAndTrim();

        System.out.println("Freezing ...");
        sample.freeze();
    }

}