package objects;

import java.util.HashMap;
import java.util.Map;

public class Puzzle {

    private final int id, nWildcards;

    private final String type;

    private final short[] initialState, currentState, solutionState;

    private final short[] tempState;

    private final int puzzleLen;

    private static short[] initializeAsShortArray(final String[] state, final Map<String, Integer> symbol_number) {
        final int n = symbol_number.size();
        final short[] values = new short[n];
        for (int i = 0; i < n; i++)
            values[i] = symbol_number.get(state[i]).shortValue();
        return values;
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

        this.initialState = initializeAsShortArray(initialState, symbol_number);
        this.currentState = initializeAsShortArray(initialState, symbol_number);
        this.solutionState = initializeAsShortArray(solutionState, symbol_number);

        this.puzzleLen = this.initialState.length;
        this.tempState = new short[puzzleLen];
    }

    public final int getID() {
        return id;
    }

    public final String getType() {
        return type;
    }

    public final short[] getInitialState() {
        return initialState;
    }

    public final short[] getCurrentState() {
        return currentState;
    }

    public final short[] getSolutionState() {
        return solutionState;
    }

    public final int getNumWildcards() {
        return nWildcards;
    }

    public final void applyMove(final int[] move) {
        int i;
        for (i = 0; i < this.puzzleLen; i++)
            this.tempState[i] = this.currentState[move[i]]; // save in temp
        for (i = 0; i < this.puzzleLen; i++) {
            this.currentState[i] = this.tempState[i]; // transfer
            this.tempState[i] = 0; // reset
        }
    }

    public final void reset() {
        System.arraycopy(this.initialState, 0, this.currentState, 0, this.puzzleLen);
    }

}
