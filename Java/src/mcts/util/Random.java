package mcts.util;

import java.util.*;
import java.util.stream.IntStream;

public class Random {

    private static final java.util.Random random = new java.util.Random();

    public static int multinomial(final double[] probabilities) {
        final double r = random.nextDouble();
        double cumProb = 0d;
        int i = 0;
        for (; i < probabilities.length; ++i)
            if ((cumProb += probabilities[i]) >= r)
                return i;
        return Math.min(i, probabilities.length - 1);
    }

    public static List<Integer> choice(final double[] probabilities, final int k) {
        final int n = probabilities.length;
        assert k <= n && n > 0;
        if (k == 0)
            return new ArrayList<>();
        if (k == n) {
            final List<Integer> allIndices = new ArrayList<>();
            for (int i = 0; i < n; ++i)
                allIndices.add(i);
            Collections.shuffle(allIndices);
            return allIndices;
        }
        if (k == 1)
            return List.of(multinomial(probabilities));

        final Map<Double, Integer> keys_indices = new TreeMap<>();
        IntStream.range(0, n).forEach(i -> {
            if (probabilities[i] == 0d)
                keys_indices.put(Double.POSITIVE_INFINITY, i);
            else
                keys_indices.put(-Math.log(random.nextDouble()) / probabilities[i], i);
        });
//        for (int i = 0; i < n; ++i) {
//            if (probabilities[i] == 0d)
//                keys_indices.put(Double.POSITIVE_INFINITY, i);
//            else
//                keys_indices.put(-Math.log(random.nextDouble()) / probabilities[i], i);
//        }

        final List<Integer> chosenIndices = new ArrayList<>();
        for (final Integer index : keys_indices.values()) {
            chosenIndices.add(index);
            if (chosenIndices.size() == k)
                break;
        }

        return chosenIndices;

//        double totalProbability = 0d;
//        for (double probability : probabilities)
//            totalProbability += probability;
//
//        final Set<Integer> chosenIndices = new HashSet<>(); // add to set because numerical instability may lead to duplicates
//        double r, cumProb;
//        int selectedIndex;
//        while (chosenIndices.size() < k) {
//            r = random.nextDouble(totalProbability);
//            cumProb = 0d;
//            for (selectedIndex = 0; selectedIndex < n; ++selectedIndex)
//                if ((cumProb += probabilities[selectedIndex]) >= r)
//                    break;
//
//            selectedIndex = Math.min(selectedIndex, n - 1); // ensure within bounds
//            chosenIndices.add(selectedIndex);
//
//            totalProbability -= probabilities[selectedIndex];
//            probabilities[selectedIndex] = 0d;
//        }
//
//        return new ArrayList<>(chosenIndices);
    }

}
