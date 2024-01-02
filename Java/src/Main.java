import data.DataLoader;
import mcts.util.Multinomial;
import objects.MoveSet;
import objects.Puzzle;

import java.io.IOException;
import java.util.*;

public class Main {

    public static void main(String... args) throws IOException {
        final Random random = new Random();
        final List<String> greetings = Arrays.asList("hi", "bye", "later");
        int idx;
        for (int i = 0; i < 1_000_000; i++) {
            idx = random.nextInt(greetings.size());
            System.out.print(idx + ": ");
            System.out.println(greetings.get(idx));
        }
    }

}
