package org.example;

/**
 * Limpieza causal del flujo de clics de UN jugador (mismo algoritmo que ml/src/data.py
 * clean_click_stream; los tests comparan ambas implementaciones con fixtures).
 *
 * 1. Solo el swing (PlayerAnimationEvent ARM_SWING) es un clic; los PlayerInteractEvent
 *    LEFT_CLICK_BLOCK y las roturas de bloque solo marcan el inicio de una excavacion.
 * 2. Un swing a menos de DEDUP_MS del ultimo swing aceptado es un duplicado (se cuenta y se descarta).
 * 3. Boton mantenido (minar): una marca de excavacion abre un "dig"; mientras el boton sigue pulsado el
 *    cliente envia un swing por tick, asi que los swings separados <= HELD_MAX_IV_MS se marcan como
 *    "held" (generados por el juego, no por el jugador) y no se evaluan como clics.
 *
 * No es thread-safe: se usa una instancia por jugador desde el hilo principal.
 */
public final class ClickStreamCleaner {

    public static final double DEDUP_MS = 15.0;
    public static final double HELD_MAX_IV_MS = 65.0;
    public static final double FRESH_DIG_MS = 15.0;

    /** Resultado de procesar un swing. */
    public static final class Swing {
        public final double tMs;
        /** Intervalo con el swing aceptado anterior (NaN para el primero). */
        public final double ivMs;
        public final boolean held;
        /** Duplicados absorbidos por este swing (se incrementa despues de crearlo). */
        public int nDup;

        Swing(double tMs, double ivMs, boolean held) {
            this.tMs = tMs;
            this.ivMs = ivMs;
            this.held = held;
        }
    }

    private boolean digging = false;
    private boolean prevHeld = false;
    private double lastDig = Double.NEGATIVE_INFINITY;
    private Swing last = null;

    /** LEFT_CLICK_BLOCK o rotura de bloque. */
    public void onDigMarker(double tMs) {
        digging = true;
        lastDig = tMs;
    }

    /**
     * Procesa un swing.
     * @return el swing aceptado, o null si es un duplicado (absorbido por el ultimo aceptado)
     */
    public Swing onSwing(double tMs) {
        if (last != null && tMs - last.tMs < DEDUP_MS) {
            last.nDup++;
            return null;
        }
        double iv = last == null ? Double.NaN : tMs - last.tMs;
        boolean fresh = (tMs - lastDig) <= FRESH_DIG_MS;
        boolean held = digging && !Double.isNaN(iv) && iv <= HELD_MAX_IV_MS && (!fresh || prevHeld);
        if (!held && !fresh) {
            digging = false;
        }
        prevHeld = held;
        last = new Swing(tMs, iv, held);
        return last;
    }
}
