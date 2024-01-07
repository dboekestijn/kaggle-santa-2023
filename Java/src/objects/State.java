package objects;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class State {

    private static final Map<Integer, short[]> size_tempValue = new HashMap<>();

    private final short[] value;

    private final int size;

    public State(final short[] value) {
        this.value = value;
        size = value.length;

        if (!size_tempValue.containsKey(size))
            size_tempValue.put(size, new short[size]);
    }

    public final void setValue(final short[] value) {
        System.arraycopy(value, 0, this.value, 0, size);
    }

    public final short[] getValue() {
        return value;
    }

    public final short get(final int i) {
        return value[i];
    }

    public final State applyMove(final Move move) {
        final short[] newValue = new short[value.length];
        for (int i = 0; i < value.length; i++)
            newValue[i] = value[move.get(i)];
        return new State(newValue);
    }

    public final State applyMoves(final List<Move> moves) {
        final short[] newValue = Arrays.copyOf(value, value.length);
        final short[] tempValue = new short[value.length];
        int i;
        for (final Move move : moves) {
            i = -1;
            while (++i < value.length)
                tempValue[i] = newValue[move.get(i)];
            while (--i >= 0)
                newValue[i] = tempValue[i];
        }

        return new State(newValue);
    }

    public final void applyMoveInplace(final int[] move) {
        final short[] tempValue = size_tempValue.get(size);
        int i = -1;
        while (++i < size)
            tempValue[i] = value[move[i]]; // save in temp
        while (--i >= 0)
            value[i] = tempValue[i]; // transfer
    }

    public final int size() {
        return size;
    }

    public final int nMatchingFacelets(final State other, final int nWildcards) {
        if (this == other)
            return size;
        if (other == null)
            return 0;
        int nMatch = 0;
        for (int i = 0; i < size; i++)
            if (value[i] == other.value[i])
                ++nMatch;
        return Math.min(nMatch + nWildcards, size);
    }

    public final boolean equalsUpTo(final State other, final int nWildcards) {
        if (this == other)
            return true;
        if (other == null)
            return false;
        if (size != other.size)
            return false;
        int nNoMatch = 0;
        for (int i = 0; i < size; i++)
            if (value[i] != other.value[i])
                if (++nNoMatch > nWildcards)
                    return false;
        return true;
    }

    public final State copy() {
        return new State(Arrays.copyOf(this.value, this.value.length));
    }

    @Override
    public final String toString() {
        return Arrays.toString(value);
    }

    @Override
    public final boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        return Arrays.equals(value, ((State) o).value);
    }

    @Override
    public final int hashCode() {
        return Arrays.hashCode(value);
    }
    
}
