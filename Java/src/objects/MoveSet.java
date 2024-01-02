package objects;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class MoveSet {

    private static final Random random = new Random();

    private final String type;

    private final Map<String, int[]> moves;

    public MoveSet(final String type, final Map<String, int[]> moves) {
        this.type = type;
        this.moves = moves;
    }

    public final String getType() {
        return type;
    }

    public final Collection<String> getMoveNames() {
        return moves.keySet();
    }

    public final Map<String, int[]> getMoves() {
        return moves;
    }

    public final int[] getMove(final String moveName) {
        return moves.get(moveName);
    }

    public final int size() {
        return moves.size();
    }

}
