package data;

import objects.Move;
import objects.MoveSet;
import objects.Puzzle;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DataLoader {

    public static final Path parent_path = Path.of("").toAbsolutePath().getParent();

    public static final Path data_dir = parent_path.resolve("data");

    public static final Path submissions_path = parent_path.resolve("submissions");

    public static Puzzle loadPuzzle(int puzzleID) throws IOException {
        try (final BufferedReader reader = Files.newBufferedReader(PuzzlesFile.path)) {
            String line = reader.readLine(); // skip header
            String[] values;
            int id;
            while ((line = reader.readLine()) != null) {
                values = line.split(PuzzlesFile.csv_sep);
                id = PuzzlesFile.getID(values);
                if (id != puzzleID)
                    continue;

                return new Puzzle(
                        id,
                        PuzzlesFile.getType(values),
                        PuzzlesFile.getInitialState(values),
                        PuzzlesFile.getSolutionState(values),
                        PuzzlesFile.getNumWildcards(values)
                );
            }
        }

        return null; // no puzzle with specified id
    }

    public static List<Puzzle> loadPuzzles() throws IOException {
        try (final BufferedReader reader = Files.newBufferedReader(PuzzlesFile.path)) {
            String line = reader.readLine(); // skip header
            String[] values;
            final List<Puzzle> puzzles = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                values = line.split(PuzzlesFile.csv_sep);
                puzzles.add(new Puzzle(
                    PuzzlesFile.getID(values),
                    PuzzlesFile.getType(values),
                    PuzzlesFile.getInitialState(values),
                    PuzzlesFile.getSolutionState(values),
                    PuzzlesFile.getNumWildcards(values)
                ));
            }

            return puzzles;
        }
    }

    public static MoveSet loadMoveSet(final String puzzleType) throws IOException {
        try (final BufferedReader reader = Files.newBufferedReader(PuzzleInfoFile.path)) {
            String line = reader.readLine(); // skip header
            String[] values;
            String type;
            while ((line = reader.readLine()) != null) {
                values = line.split(PuzzleInfoFile.csv_sep);
                type = PuzzleInfoFile.getType(values);
                if (!type.equals(puzzleType))
                    continue;

                return new MoveSet(type, PuzzleInfoFile.getMoves(values));
            }
        }

        return null; // no move set with specified type
    }

    public static Map<String, MoveSet> loadMoveSets() throws IOException {
        try (final BufferedReader reader = Files.newBufferedReader(PuzzleInfoFile.path)) {
            String line = reader.readLine(); // skip header
            String[] values;
            String type;
            final Map<String, MoveSet> type_moveSet = new HashMap<>();
            while ((line = reader.readLine()) != null) {
                values = line.split(PuzzleInfoFile.csv_sep);
                type = PuzzleInfoFile.getType(values);
                type_moveSet.put(
                    type,
                    new MoveSet(
                        PuzzleInfoFile.getType(values),
                        PuzzleInfoFile.getMoves(values)
                    )
                );
            }

            return type_moveSet;
        }
    }

    public static List<Move> loadSampleSubmissionMoves(final int puzzleID, final MoveSet moveSet) throws IOException {
        try (final BufferedReader reader = Files.newBufferedReader(SampleSubmissionFile.path)) {
            String line = reader.readLine(); // skip header
            String[] values;
            int id;
            while ((line = reader.readLine()) != null) {
                values = line.split(SampleSubmissionFile.csv_sep);
                id = SampleSubmissionFile.getID(values);
                if (id != puzzleID)
                    continue;

                final List<Move> movesList = new ArrayList<>();
                for (final String moveName : SampleSubmissionFile.getMoveNames(values))
                    movesList.add(moveSet.getMove(moveName));
                return movesList;
            }
        }

        return null; // no puzzle with specified id
    }

    public static Map<Integer, List<Move>> loadSampleSubmissionMoves(final MoveSet moveSet) throws IOException {
        try (final BufferedReader reader = Files.newBufferedReader(SampleSubmissionFile.path)) {
            String line = reader.readLine(); // skip header
            String[] values;
            final Map<Integer, List<Move>> puzzleID_movesList = new HashMap<>();
            List<Move> movesList;
            while ((line = reader.readLine()) != null) {
                values = line.split(SampleSubmissionFile.csv_sep);
                movesList = new ArrayList<>();
                for (final String moveName : SampleSubmissionFile.getMoveNames(values))
                    movesList.add(moveSet.getMove(moveName));

                puzzleID_movesList.put(SampleSubmissionFile.getID(values), movesList);
            }

            return puzzleID_movesList;
        }
    }

}
