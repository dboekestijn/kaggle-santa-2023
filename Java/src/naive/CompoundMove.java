package naive;

import java.util.Arrays;

public record CompoundMove(String name, short[] value) {

    public short get(final int i) {
        return value[i];
    }

    public int level() {
        return name.split(";").length;
    }

    public int size() {
        return value.length;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder(name + ",");
        for (final short v : value)
            sb.append(v).append(";");
        return sb.substring(0, sb.length() - 1);
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o)
            return true;
        if (o == null || !getClass().equals(o.getClass()))
            return false;
        return Arrays.equals(value, ((CompoundMove) o).value);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(value);
    }

}
