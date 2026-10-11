package org.example;

/**
 * Regla de FastBreak basada en el tiempo esperado de rotura de cada bloque.
 *
 * El servidor conoce la velocidad de rotura (Block#getBreakSpeed(Player): progreso por tick segun
 * bloque, herramienta, encantamientos, efectos, agua y si el jugador esta en el suelo). Un bloque se
 * rompe cuando el progreso acumulado llega a 1, es decir en ceil(1 / velocidad) ticks. El propio
 * servidor vanilla acepta la rotura con un progreso >= 0.7, asi que solo se marca una rotura que
 * llega antes de ese margen. Codigo puro (sin Bukkit) para poder probarlo con tests.
 */
public final class FastBreakCheck {

    public static final double TICK_MS = 50.0;

    private FastBreakCheck() { }

    /** Tiempo esperado (ms) para romper un bloque con esa velocidad; 0 si es instantaneo. */
    public static double expectedMs(float speedPerTick) {
        if (speedPerTick >= 1.0f) return 0.0;
        if (speedPerTick <= 0.0f) return Double.POSITIVE_INFINITY; // irrompible
        return Math.ceil(1.0 / speedPerTick) * TICK_MS;
    }

    /**
     * @param elapsedMs  tiempo entre el inicio de la excavacion (LEFT_CLICK_BLOCK) y la rotura
     * @param speedStart velocidad al empezar a picar
     * @param speedEnd   velocidad en el momento de la rotura
     * @param tolerance  fraccion minima del tiempo esperado que debe transcurrir (0.7 como vanilla)
     */
    public static boolean tooFast(double elapsedMs, float speedStart, float speedEnd, double tolerance) {
        // Se usa la velocidad mas favorable al jugador: si salta o entra en agua a mitad, no se le penaliza
        double expected = expectedMs(Math.max(speedStart, speedEnd));
        if (expected <= 0.0 || Double.isInfinite(expected)) return false;
        return elapsedMs < expected * tolerance;
    }
}
