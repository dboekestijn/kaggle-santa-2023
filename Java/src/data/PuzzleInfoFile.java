package data;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PuzzleInfoFile {

    public static final Path path = DataLoader.data_dir.resolve("puzzle_info.csv");

    public static final String csv_sep = ",(?=\\\")";

    public static final int type_col = 0, moves_col = 1;

    private static final Pattern move_regex_pattern =
            Pattern.compile("'(\\w+)': \\[((?:\\d+(?:, )*)+)\\]", Pattern.MULTILINE);

    public static String getType(final String[] values) {
        return values[type_col];
    }

    public static Map<String, int[]> getMoves(final String[] values) {
        final String moves = values[moves_col];
        final Matcher matcher = move_regex_pattern.matcher(moves);

        final Map<String, int[]> moveName_moveArray = new HashMap<>();
        String moveName, move;
        String[] moveElements;
        int[] moveArray;
        int i;
        while (matcher.find()) {
            moveName = matcher.group(1);
            move = matcher.group(2);
            moveElements = move.split(", ");
            moveArray = new int[moveElements.length];
            for (i = 0; i < moveArray.length; i++)
                moveArray[i] = Short.parseShort(moveElements[i]);

            moveName_moveArray.put(moveName, moveArray);
        }

        return moveName_moveArray;
    }

}
