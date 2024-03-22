package naive;

import data.DataLoader;
import objects.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;

public class Main3 {

    public static void main(final String[] args) throws IOException, ExecutionException, InterruptedException {
        Locale.setDefault(Locale.getDefault());

        final int[] puzzleIDs = new int[] {
            0, // cube_2/2/2
            30, // cube_3/3/3
            150, // cube_4/4/4
            210, // cube_5/5/5
            245, // cube_6/6/6
            257, // cube_7/7/7
            262, // cube_8/8/8
            267, // cube_9/9/9
            272, // cube_10/10/10
            277, // cube_19/19/19
            281, // cube_33/33/33
            284, // wreath_6/6
            304, // wreath_7/7
            319, // wreath_12/12
            329, // wreath_21/21
            334, // wreath_33/33
            337, // wreath_100/100
            338, // globe_1/8
            348, // globe_1/16
            353, // globe_2/6
            358, // globe_3/4
            373, // globe_6/4
            378, // globe_6/8
            383, // globe_6/10
            388, // globe_3/33
            396  // globe_8/25
        };

        for (final int puzzleID : puzzleIDs) {
            try {
                populateStatePairs(puzzleID);
            } catch (final OutOfMemoryError e) {
                e.printStackTrace();
            }

            System.gc();
            Thread.sleep(60_000L);
        }
    }

    private static void populateStatePairs(final int puzzleID) throws IOException {
        /* --- LOAD DATA --- */
        // load puzzle
        final Puzzle puzzle = DataLoader.loadPuzzle(puzzleID);
        assert puzzle != null;

        // load corresponding move set
        final MoveSet moveSet = DataLoader.loadMoveSet(puzzle.type);
        assert moveSet != null;

        System.out.printf("Move set for puzzle type '%s' has branching factor %d.\n",
                puzzle.type, moveSet.size());

        /* CREATING COMPOUND MOVE SET */
        final int maxCompoundMoveDepth = 100;
        final Set<objects.CompressedMove> compressedMoveSet = getCompressedMoveSet(moveSet, maxCompoundMoveDepth);
        System.out.println("Found " + compressedMoveSet.size() + " unique (compound) moves.");
        System.gc();

        /* --- WRITING MOVE PAIRS --- */
        final Path writeDir = Path.of("")
                .resolve("data")
                .resolve(puzzle.type.replace("/", "-"))
//                .resolve("puzzle_" + puzzleID)
                .toAbsolutePath();
        final File writeDirFile = writeDir.toFile();
        if (!writeDirFile.exists() && !writeDir.toFile().mkdirs())
            throw new IOException("dir creation failed");

        final Path writePath = writeDir.resolve("move_data.csv");
        writeMoves(compressedMoveSet, writePath);
    }

    private static Set<CompressedMove> getCompressedMoveSet(final MoveSet moves, final int maxDepth) {
        Set<CompressedMove> pastMoveSet = new HashSet<>();
        for (final Move move : moves.getMoves())
            pastMoveSet.add(new CompressedMove(move.name, Arrays.copyOf(move.value, move.value.length)));

        final Set<CompressedMove> completeMoveSet = new HashSet<>(pastMoveSet);
        Set<CompressedMove> newMoveSet;
        CompressedMove newMove;
        for (int depth = 2; depth <= maxDepth; ++depth) {
            System.out.println("At depth " + depth + " ...");

            newMoveSet = new HashSet<>();
            for (final CompressedMove pastMove : pastMoveSet)
                for (final Move move : moves.getMoves()) {
                    try {
                        newMove = pastMove.applyMove(move);
                        if (!completeMoveSet.contains(newMove)) {
                            newMoveSet.add(newMove);
                            completeMoveSet.add(newMove);
                        } else
                            newMove = null; // for gc
                    } catch (final OutOfMemoryError e) {
                        return completeMoveSet;
                    }
                }

            if (newMoveSet.isEmpty())
                break;

            pastMoveSet = newMoveSet;
            newMoveSet = null; // for gc
            System.gc();
        }

        return completeMoveSet;
    }

//    private static CompoundMove getNewCompoundMove(final CompoundMove compoundMove, final Move move) {
//        final short[] newMove = new short[compoundMove.size()];
//        for (int i = 0; i < newMove.length; ++i)
//            newMove[i] = compoundMove.get(move.get(i));
//        return new CompoundMove(compoundMove.name() + ";" + move.name, newMove);
//    }

    private static long getMemoryUsage() {
        return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
    }

    private static void writeMoves(final Set<CompressedMove> moves, final Path writePath) throws IOException {
        try (final BufferedWriter writer = Files.newBufferedWriter(writePath,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)) {
            writer.write("name");
            for (final CompressedMove move : moves) {
                writer.newLine();
                writer.write(move.name());
            }
        }
    }

//    private static Map<String, SimpleMove> getCompoundMoveMap(final MoveSet moveSet, final int maxDepth) throws
//            InterruptedException, ExecutionException {
//        final List<Callable<Set<List<String>>>> generators = new ArrayList<>();
//        for (final Move move : moveSet.getMoves()) {
//            generators.add(() -> {
//                final Set<List<String>> moveSequences = new HashSet<>();
//                generateCompoundMoves(moveSequences, moveSet, new ArrayList<>(), move.name, 1, maxDepth);
//                return moveSequences;
//            });
//        }
//
//        final Set<List<String>> allMoveSequences = new HashSet<>();
//        for (final Future<Set<List<String>>> result : executor.invokeAll(generators))
//            allMoveSequences.addAll(result.get());
//
//        final Set<SimpleMove> compoundMoveSet = new HashSet<>();
//
//        SimpleMove compoundMove;
//        for (final List<String> moveSequence : allMoveSequences) {
//            compoundMove = createNewMove(moveSequence.stream().map(moveSet::getMove).toList());
//            compoundMoveSet.add(compoundMove);
//        }
//
//        final Map<String, SimpleMove> compoundMoveMap = new HashMap<>();
//        compoundMoveSet.forEach(move -> compoundMoveMap.put(move.name, move));
//        return compoundMoveMap;
//    }

//    private static void generateCompoundMoves(final Set<List<String>> allMoveSequences, final MoveSet moveSet,
//                                              final List<String> currentMoveSequence, final String moveName,
//                                              final int depth, final int maxDepth) {
//        currentMoveSequence.add(moveName);
//        allMoveSequences.add(currentMoveSequence);
//        if (depth == maxDepth)
//            return;
//
//        for (final Move move : moveSet.getMoves())
//            if (!currentMoveSequence.contains(move.name))
//                generateCompoundMoves(allMoveSequences, moveSet,
//                        new ArrayList<>(currentMoveSequence), move.name,
//                        depth + 1, maxDepth);
//    }

    private static CompressedMove createNewMove(final List<Move> moves) {
        final Move fstMove = moves.get(0);
        final short[] compoundMove = Arrays.copyOf(fstMove.value, fstMove.value.length);
        final short[] temp = new short[compoundMove.length];

        final StringBuilder compoundMoveNameBuilder = new StringBuilder(fstMove.name);
        int i;
        for (final Move move : moves.subList(1, moves.size())) {
            compoundMoveNameBuilder.append(";").append(move.name);
            i = -1;
            while (++i < compoundMove.length)
                temp[i] = compoundMove[move.get(i)];
            while (--i >= 0)
                compoundMove[i] = temp[i];
        }

        return new CompressedMove(compoundMoveNameBuilder.toString(), compoundMove);
    }

    private static void writeStatePairs(final Puzzle puzzle, final MoveSet moveSet,
                                        final int nPairs, final Path writePath) throws IOException {
        System.out.println("Generating reachable states ...");
        final long t0 = System.currentTimeMillis();
        final Set<State> reachableStates =
                new HashSet<>(moveSet.getMoves().stream().map(puzzle.solutionState::applyMove).toList());

        final int nStates = (int) Math.ceil(nPairs / (double) moveSet.size());
        Set<State> newStates;
        State newState;
        boolean breakOut;
        do {
            newStates = new HashSet<>();
            breakOut = false;
            for (final State state : reachableStates) {
                for (final Move move : moveSet.getMoves()) {
                    newState = state.applyMove(move);
                    if (!reachableStates.contains(newState))
                        newStates.add(newState);
                    if (newStates.size() + reachableStates.size() == nStates) {
                        breakOut = true;
                        break;
                    }
                }
                if (breakOut)
                    break;
            }
            if (newStates.isEmpty())
                break;

            reachableStates.addAll(newStates);
        } while (reachableStates.size() < nStates);

        final long dt = System.currentTimeMillis() - t0;
        System.out.printf("Found %d unique reachable states in %d s.\n", reachableStates.size(), dt / 1_000);
        System.out.println();

        System.out.println("Writing state pairs ...");
        final boolean exists = Files.exists(writePath);

        try (final BufferedWriter writer = Files.newBufferedWriter(writePath,
                exists ? StandardOpenOption.APPEND : StandardOpenOption.CREATE_NEW)) {
            if (!exists)
                writer.write("input,target");

            State initialState;
            int[] inputState;
            for (final State solutionState : reachableStates) {
                for (final Move move : moveSet.getMoves()) {
                    initialState = solutionState.applyMove(move);
                    inputState = new int[initialState.size()];
                    for (int i = 0; i < inputState.length; ++i)
                        inputState[i] = initialState.get(i) == solutionState.get(i) ? 1 : 0;

                    writer.newLine();
                    writer.write(String.join(",",
                        Arrays.toString(inputState),
                        move.name
                    ));
                }
            }
        }
    }

}
