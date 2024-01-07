package objects;

import java.util.*;

public class MoveSet {

    private final String type;

    private final Map<String, Move> moves;

    public MoveSet(final String type, final Map<String, int[]> moves) {
        this.type = type;
        this.moves = new HashMap<>();
        Move move, reverseMove;
        for (Map.Entry<String, int[]> moveEntry : moves.entrySet()) {
            move = new Move(moveEntry.getKey(), moveEntry.getValue());
            this.moves.put(move.name, move);

            reverseMove = move.getReverse();
            this.moves.put(reverseMove.name, reverseMove);
        }
    }

    public final String getType() {
        return type;
    }

    public final Map<String, Move> getMoveMap() {
        return moves;
    }

    public final Collection<Move> getMoves() {
        return moves.values();
    }

    public final Move getMove(final String moveName) {
        return moves.get(moveName);
    }

    public final int size() {
        return moves.size();
    }

}
