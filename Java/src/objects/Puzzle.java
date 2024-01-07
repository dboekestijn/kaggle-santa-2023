package objects;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class Puzzle {

    public final int id, nWildcards;

    public final String type;

    public final State initialState, solutionState;

    public State currentState;

    private int defaultLenShortestPath;

    private static State initializeAsState(final String[] state, final Map<String, Integer> symbol_number) {
        final int n = state.length;
        final short[] values = new short[n];
        for (int i = 0; i < n; i++)
            values[i] = symbol_number.get(state[i]).shortValue();
        return new State(values);
    }

    public Puzzle(final int id,
                  final String type,
                  final String[] initialState,
                  final String[] solutionState,
                  final int nWildcards) {
        this.id = id;
        this.type = type;
        this.nWildcards = nWildcards;

        final Map<String, Integer> symbol_number = new HashMap<>();
        Integer number = -1;
        for (final String symbol : solutionState) {
            number = symbol_number.getOrDefault(symbol, number + 1);
            symbol_number.put(symbol, number);
        }

        this.initialState = initializeAsState(initialState, symbol_number);
        this.currentState = initializeAsState(initialState, symbol_number);
        this.solutionState = initializeAsState(solutionState, symbol_number);

        defaultLenShortestPath = Integer.MAX_VALUE;
    }

    public Puzzle(final int id,
                  final String type,
                  final State initialState,
                  final State solutionState,
                  final int nWildcards) {
        this.id = id;
        this.type = type;
        this.initialState = initialState;
        this.currentState = initialState.copy();
        this.solutionState = solutionState;
        this.nWildcards = nWildcards;
    }

    public final int getDefaultLenShortestPath() {
        return defaultLenShortestPath;
    }

    public final void setDefaultLenShortestPath(final int len) {
        defaultLenShortestPath = len;
    }

    public final void applyMove(final Move move) {
        currentState = currentState.applyMove(move);
    }

    public final void applyMoves(final List<Move> moves) {
        moves.forEach(this::applyMove);
    }

    public final boolean isSolved() {
        return currentState.equalsUpTo(solutionState, nWildcards);
    }

    @Override
    public final boolean equals(final Object o) {
        if (this == o)
            return true;
        if (o == null || !getClass().equals(o.getClass()))
            return false;
        return id == ((Puzzle) o).id;
    }

    @Override
    public final int hashCode() {
        return Objects.hash(id);
    }

}
