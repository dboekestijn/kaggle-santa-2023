package data;

import java.nio.file.Path;

public class SampleSubmissionFile {

    public static final Path path = DataLoader.submissions_path.resolve("sample_submission.csv");

    public static final String csv_sep = ",";

    private static final int id_col = 0, moves_col = 1;

    private static final String moves_sep = "\\.";

    public static int getID(final String[] values) {
        return Integer.parseInt(values[id_col]);
    }

    public static String[] getMoveNames(final String[] values) {
        return values[moves_col].split(moves_sep);
    }

}
