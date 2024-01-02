package mcts.util;

import java.util.Arrays;

public class Softmax {

    public static double[] probabilities(final double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values)
            if (value > max)
                max = value;

        double denom = 0d;
        for (double value : values)
            denom += Math.exp(value - max);

        final double[] result = new double[values.length];
        for (int i = 0; i < result.length; i++)
            result[i] = Math.exp(values[i] - max) / denom;

        return result;
    }

}
