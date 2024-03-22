package objects;

import java.util.*;

public class MoveSet {

    private final String type;

    private final Map<String, Move> movesMap;

    private final Move[] moves;

    public MoveSet(final String type, final Map<String, int[]> moves) {
        this.type = type;
        this.movesMap = new HashMap<>();
        this.moves = new Move[2 * moves.size()];

        Move move, reverseMove;
        int[] intMove;
        short[] shortMove;
        int moveIdx = -1;
        for (Map.Entry<String, int[]> moveEntry : moves.entrySet()) {
            intMove = moveEntry.getValue();
            shortMove = new short[intMove.length];
            for (int i = 0; i < shortMove.length; ++i)
                shortMove[i] = (short) intMove[i];
            move = new Move(moveEntry.getKey(), shortMove);
            this.movesMap.put(move.name, move);
            this.moves[++moveIdx] = move;

            reverseMove = move.getReverse();
            this.movesMap.put(reverseMove.name, reverseMove);
            this.moves[++moveIdx] = reverseMove;
        }
    }

    public final String getType() {
        return type;
    }

    public final Map<String, Move> getMovesMap() {
        return movesMap;
    }

    public final Collection<Move> getMoves() {
        return movesMap.values();
    }

    public final Move getMove(final String moveName) {
        return movesMap.get(moveName);
    }

    public final Move getMove(final int moveIdx) {
        return moves[moveIdx];
    }

    public final int size() {
        return movesMap.size();
    }

}
