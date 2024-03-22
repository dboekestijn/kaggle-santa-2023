package bnb;

import java.util.Arrays;

public class Vertex {

    public final short[] state;

    public Vertex(final short[] state) {
        this.state = state;
    }

    @Override
    public final boolean equals(final Object o) {
        if (this == o)
            return true;
        if (o == null || !getClass().equals(o.getClass()))
            return false;
        return Arrays.equals(state, ((Vertex) o).state);
    }

    @Override
    public final int hashCode() {
        return Arrays.hashCode(state);
    }

}
