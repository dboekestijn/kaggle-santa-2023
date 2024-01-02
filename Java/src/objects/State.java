package objects;

import java.util.Arrays;

public class State {

    public static State of(final short[] value) {
        return new State(value);
    }

    public final short[] value;

    public State(final short[] value) {
        this.value = value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        return Arrays.equals(value, ((State) o).value);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(value);
    }
    
}
