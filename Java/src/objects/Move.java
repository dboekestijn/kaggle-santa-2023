package objects;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;

public class Move {

    private static final String reverse_move_prefix = "-";

    public final String name;

    public final int[] value;

    public Move(final String name, final int[] value) {
        this.name = name;
        this.value = value;
    }

    public final int get(final int i) {
        return value[i];
    }

    public final Move getReverse() {
        final Integer[] indices = new Integer[value.length];
        int i = -1;
        while (++i < value.length)
            indices[i] = i;

        Arrays.sort(indices, Comparator.comparingInt(idx -> value[idx]));

        final int[] result = new int[value.length];
        while (--i >= 0)
            result[i] = indices[i];

        return new Move(reverse_move_prefix + name, result);
    }

    public final boolean isReverse() {
        return name.startsWith(reverse_move_prefix);
    }

    @Override
    public final String toString() {
        return "Move<" + name + ", " + Arrays.toString(value) + ">";
    }

    @Override
    public final boolean equals(final Object o) {
        if (this == o)
            return true;
        if (o == null || !getClass().equals(o.getClass()))
            return false;
        return name.equals(((Move) o).name);
    }

    @Override
    public final int hashCode() {
        return Objects.hash(name);
    }

}
