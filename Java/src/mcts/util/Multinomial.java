package mcts.util;

import java.util.Random;

public class Multinomial {

    private static final Random random = new Random();

    public static int draw(final double[] probabilities) {
        final double r = random.nextDouble();
        double cum_prod = 0d;
        int i = 0;
        for (; i < probabilities.length; i++)
            if ((cum_prod += probabilities[i]) >= r)
                return i;
        return i;
    }

}
