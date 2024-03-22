package quarantine;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class HashedMove {

    private static final int hash_scale = 100;

    private static final RoundingMode hash_rounding = RoundingMode.HALF_EVEN;

    private static final BigDecimal hash_diff = BigDecimal.valueOf(0.1).setScale(hash_scale, hash_rounding);

    private static final BigDecimal hash_base = BigDecimal.ONE.add(hash_diff).setScale(hash_scale, hash_rounding);

    private static BigDecimal getSlope(final short len) {
        return hash_diff.divide(BigDecimal.valueOf(len + 1), hash_rounding);
    }

    private static BigDecimal transformValue(final BigDecimal slope, final BigDecimal value) {
        return BigDecimal.ONE.add(slope.multiply(value)); // f(x_i) = 1 + a * x_i
    }

    private static BigDecimal getHashedValue(final BigDecimal slope, final BigDecimal originalValue, final short i) {
        return transformValue(slope, originalValue)
                .multiply(hash_base.pow(i)); // f(x_i) * B^i
    }

    private static BigDecimal getHashedValue(final BigDecimal slope, final short[] value, final short i) {
        return getHashedValue(slope, BigDecimal.valueOf(value[i]), i);
    }

    private static BigDecimal computeHash(final short[] value) {
        final BigDecimal slope = getSlope((short) value.length);
        BigDecimal hashValue = BigDecimal.ZERO.setScale(hash_scale, hash_rounding);
        for (short i = 0; i < value.length; ++i)
            // h = h + B^i * (1 + a * x_i)
            hashValue = hashValue.add(getHashedValue(slope, value, i));
        return hashValue;
    }

    private final String name;

    private final BigDecimal hashedValue;

    public HashedMove(final String name, final short[] value) {
        this.name = name;
        hashedValue = computeHash(value);
    }

    public final String name() {
        return name;
    }

    public final BigDecimal hashedValue() {
        return hashedValue;
    }

    @Override
    public final boolean equals(final Object o) {
        if (this == o)
            return true;
        if (!(o instanceof HashedMove))
            return false;
        return hashedValue.equals(((HashedMove) o).hashedValue);
    }

    @Override
    public final int hashCode() {
        return hashedValue.hashCode();
    }

}
