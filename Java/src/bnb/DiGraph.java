package bnb;

import objects.MoveSet;
import objects.Puzzle;
import objects.State;

import java.util.*;

public class DiGraph {

    private final Puzzle puzzle;

    private final MoveSet moveSet;

    private final State source, sink;

    private final Set<State> vertices;

    private final Set<Arc> arcs;

    private final Map<State, Map<State, Arc>> vFrom_vTo_arc, vTo_vFrom_arc;

    public DiGraph(final Puzzle puzzle, final MoveSet moveSet, final List<List<Integer>> movePaths) {
        this.puzzle = puzzle;
        this.moveSet = moveSet;

        source = puzzle.initialState.copy();
        sink = puzzle.solutionState.copy();
        vertices = new HashSet<>(Arrays.asList(source, sink));
        arcs = new HashSet<>();
        vFrom_vTo_arc = new HashMap<>();
        vTo_vFrom_arc = new HashMap<>();

        populateGraph(new HashSet<>(movePaths));
    }

    private void populateGraph(final Set<List<Integer>> movePaths) {
        Map<State, Arc> vTo_arc, vFrom_arc;
        State fromState, toState;
        Arc arc;
        for (final List<Integer> movePath : movePaths) {
            fromState = source;
            toState = null;
            for (final Integer moveIdx : movePath) {
                toState = fromState.applyMove(moveSet.getMove(moveIdx));
                vertices.add(toState);

                vTo_arc = vFrom_vTo_arc.getOrDefault(fromState, new HashMap<>());
                vFrom_arc = vTo_vFrom_arc.getOrDefault(toState, new HashMap<>());
                arc = vTo_arc.getOrDefault(toState, vFrom_arc.getOrDefault(fromState, new Arc(fromState, toState)));
                if (!vTo_arc.containsKey(toState))
                    vTo_arc.put(toState, arc);
                if (!vFrom_arc.containsKey(fromState))
                    vFrom_arc.put(fromState, arc);

                vFrom_vTo_arc.put(fromState, vTo_arc);
                vTo_vFrom_arc.put(toState, vFrom_arc);
                arcs.add(arc);

                fromState = toState;
            }

            assert Objects.equals(toState, sink); // last state must be the solution state
        }
    }

    public final Set<State> getVertices() {
        return vertices;
    }

    public final Arc getArc(final State from, final State to) {
        final Map<State, Arc> vTo_arc = vFrom_vTo_arc.get(from);
        if (vTo_arc == null)
            return null;
        return vTo_arc.get(to);
    }

    public final Set<Arc> getArcs() {
        return arcs;
    }

    public final Collection<Arc> getInArcs(final State to) {
        final Map<State, Arc> inArcMap = vTo_vFrom_arc.get(to);
        if (inArcMap == null)
            return null;
        return inArcMap.values();
    }

    public final Collection<Arc> getOutArcs(final State from) {
        final Map<State, Arc> outArcMap = vFrom_vTo_arc.get(from);
        if (outArcMap == null)
            return null;
        return outArcMap.values();
    }

    public void pruneGraph() {
        final Collection<Arc> inArcs = getInArcs(sink);
        if (inArcs == null)
            throw new IllegalStateException("no arcs connecting to sink state");

        for (final Arc inArc : inArcs)
            updateArcWeights(sink, inArc, 1); // in bottom-up fashion
        pruneArcs(source); // in top-down fashion

        cleanupGraph();
    }

    private void updateArcWeights(final State fromState, final Arc outArc, final int lenShortestPath) {
        outArc.updateWeight(lenShortestPath);
        if (fromState.equals(source))
            return;

        final Collection<Arc> inArcs = getInArcs(fromState);
        if (inArcs == null)
            return;

        for (final Arc inArc : inArcs)
            updateArcWeights(inArc.from, inArc, lenShortestPath + 1);
    }

    private void pruneArcs(final State fromState) {
        if (fromState.equals(sink))
            return;

        final Collection<Arc> outArcs = getOutArcs(fromState);
        if (outArcs == null || outArcs.isEmpty())
            return;

        int lenShortestPath = outArcs.stream().mapToInt(Arc::getWeight).min().orElseThrow();
        outArcs.removeIf(arc -> arc.getWeight() > lenShortestPath);
        for (final Arc outArc : outArcs)
            pruneArcs(outArc.to);
    }

    private void cleanupGraph() {
        vertices.removeIf(vertex -> {
            final Collection<Arc> inArcs = getInArcs(vertex), outArcs = getOutArcs(vertex);
            return (inArcs == null || inArcs.isEmpty()) && (outArcs == null || outArcs.isEmpty());
        });
    }

}
