package objects;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CompressedMove {

    public static short[] compressMove(final short[] value) {
        final List<Short> compressedMoveList = new ArrayList<>();
        compressedMoveList.add(value[0]);

        short compressedValue;
        boolean increasing, decreasing;
        int i, j;
        for (i = 1; i < value.length; ) {
            increasing = value[i] == value[i - 1] + 1;
            decreasing = value[i] == value[i - 1] - 1;
            if (!increasing && !decreasing) {
                compressedMoveList.add(value[i]);
                ++i;
                continue;
            }

            if (increasing) {
                for (j = i + 1; j < value.length; ++j)
                    if (value[j] != value[j - 1] + 1)
                        break;
            } else { // decreasing
                for (j = i + 1; j < value.length; ++j)
                    if (value[j] != value[j - 1] - 1)
                        break;
            }

            if (j > i + 1) {
                compressedValue = (short) -value[j - 1];
                if (compressedValue == 0)
                    compressedValue = compressed_zero;
                compressedMoveList.add(compressedValue);
            } else
                compressedMoveList.add(value[i]);
            i = j;
        }

        final short[] compressedMove = new short[compressedMoveList.size()];
        for (i = 0; i < compressedMove.length; ++i)
            compressedMove[i] = compressedMoveList.get(i);
        return compressedMove;
    }

    private static final short compressed_zero = Short.MIN_VALUE;

    private final String name;

    private final short[] compressedValue;

    private final int originalLength;

    public CompressedMove(final String name, final short[] originalValue) {
        this.name = name;
        this.compressedValue = compressMove(originalValue);
        originalLength = originalValue.length;
    }

    public final CompressedMove applyMove(final Move move) {
        final short[] newValue = new short[originalLength];
        newValue[move.getNewIndex(0)] = compressedValue[0];

        for (int c = 1, i = 1, j, jEnd; i < originalLength; ++c) {
            if ((jEnd = compressedValue[c]) < 0) {
                jEnd = jEnd == compressed_zero ? 0 : -jEnd;
                j = compressedValue[c - 1];
                --i; // would have been added in previous iteration (and starts at 1)
                if (jEnd > j) { // increasing
                    for ( ; j <= jEnd; ++j, ++i)
                        newValue[move.getNewIndex(i)] = (short) j;
                } else { // decreasing
                    for ( ; j >= jEnd; --j, ++i)
                        newValue[move.getNewIndex(i)] = (short) j;
                }
            } else
                newValue[move.getNewIndex(i++)] = compressedValue[c];
        }

        return new CompressedMove(name + ";" + move.name, newValue);
    }

    public final String name() {
        return name;
    }

    public final short[] compressedValue() {
        return compressedValue;
    }

    public final short[] expandedValue() {
        final short[] expandedValue = new short[originalLength];
        expandedValue[0] = compressedValue[0];

        for (int c = 1, i = 1, j, jEnd; i < originalLength; ++c) {
            if ((jEnd = compressedValue[c]) < 0) {
                jEnd = jEnd == compressed_zero ? 0 : -jEnd;
                j = compressedValue[c - 1];
                --i; // would have been added in previous iteration (and starts at 1)
                if (jEnd > j) { // increasing
                    for ( ; j <= jEnd; ++j, ++i)
                        expandedValue[i] = (short) j;
                } else { // decreasing
                    for ( ; j >= jEnd; --j, ++i)
                        expandedValue[i] = (short) j;
                }
            } else
                expandedValue[i++] = compressedValue[c];
        }

        return expandedValue;
    }

    public final int originalLength() {
        return originalLength;
    }

    @Override
    public final String toString() {
        return name + "," + Arrays.toString(compressedValue);
    }

    @Override
    public final boolean equals(final Object o) {
        if (this == o)
            return true;
        if (!(o instanceof CompressedMove))
            return false;
        return Arrays.equals(compressedValue, ((CompressedMove) o).compressedValue);
    }

    @Override
    public final int hashCode() {
        return Arrays.hashCode(compressedValue);
    }

}
