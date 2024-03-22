package objects;

import java.util.*;

public class Move {

    private static final String reverse_move_prefix = "-";

    public final String name;

    public final short[] value;

    private final short[] toValue;

    public Move(final String name, final short[] value) {
        this.name = name;
        this.value = value;

        toValue = new short[value.length];
        for (short i = 0; i < value.length; ++i)
            toValue[value[i]] = i;
    }

    public final int get(final int i) {
        return value[i];
    }

    public final int getNewIndex(final int i) {
        return toValue[i];
    }

    public final Move getReverse() {
        final Short[] indices = new Short[value.length];
        short i = -1;
        while (++i < value.length)
            indices[i] = i;
        Arrays.sort(indices, Comparator.comparingInt(idx -> value[idx]));

        final short[] result = new short[value.length];
        while (--i >= 0)
            result[i] = indices[i];

        return new Move(reverse_move_prefix + name, result);
    }

    public final String toSimpleString() {
        final StringBuilder sb = new StringBuilder();
        for (final short v : value)
            sb.append(";").append(v);
        return sb.substring(1);
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
