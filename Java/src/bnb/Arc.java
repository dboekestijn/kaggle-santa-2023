package bnb;

import objects.State;

import java.util.Objects;

public class Arc {

    public final State from, to;

    private int weight;

    public Arc(final State from, final State to) {
        this(from, to, Integer.MAX_VALUE);
    }

    public Arc(final State from, final State to, final int weight) {
        this.from = from;
        this.to = to;
        this.weight = weight;
    }

    public final void updateWeight(final int weight) {
        this.weight = Math.min(this.weight, weight);
    }

    public final int getWeight() {
        return weight;
    }

    @Override
    public final boolean equals(final Object o) {
        if (this == o)
            return true;
        if (o == null || !getClass().equals(o.getClass()))
            return false;
        final Arc other = (Arc) o;
        return Objects.equals(from, other.from) && Objects.equals(to, other.to);
    }

    @Override
    public final int hashCode() {
        return Objects.hash(from, to);
    }

}
