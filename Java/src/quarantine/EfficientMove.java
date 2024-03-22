package quarantine;

import java.util.Arrays;

public record EfficientMove(byte[] moves, short[] value) {

    public int size() {
        return value.length;
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o)
            return true;
        if (o == null || !getClass().equals(o.getClass()))
            return false;
        return Arrays.equals(value, ((EfficientMove) o).value);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(value);
    }

}
