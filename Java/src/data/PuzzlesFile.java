package data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class PuzzlesFile {

    public static final Path path = DataLoader.data_dir.resolve("puzzles.csv");

    public static final String csv_sep = ",";

    private static final int id_col = 0, type_col = 1, sol_col = 2, init_col = 3, wildcards_col = 4;

    private static final String array_sep = ";";

    public static int getID(final String[] values) {
        return Integer.parseInt(values[id_col]);
    }

    public static String getType(final String[] values) {
        return values[type_col];
    }

    public static String[] getSolutionState(final String[] values) {
        return values[sol_col].split(array_sep);
    }

    public static String[] getInitialState(final String[] values) {
        return values[init_col].split(array_sep);
    }

    public static int getNumWildcards(final String[] values) {
        return Integer.parseInt(values[wildcards_col]);
    }

}
