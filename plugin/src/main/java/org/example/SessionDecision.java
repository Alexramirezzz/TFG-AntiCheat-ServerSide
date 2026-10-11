package org.example;

/**
 * Decision por sesion: alarma cuando al menos K de las ultimas N ventanas evaluadas estan marcadas
 * (igual que session_alarms en ml/src/decision.py). Acumular evidencia antes de sumar VL reduce los
 * falsos positivos frente a sancionar ventana a ventana. Tras una alarma el historial se reinicia, de
 * modo que cada VL exige K ventanas marcadas nuevas.
 *
 * No es thread-safe: una instancia por jugador, usada desde el hilo principal.
 */
public final class SessionDecision {

    private final int k;
    private final boolean[] ring;
    private int pos = 0;
    private int count = 0;   // ventanas marcadas en el anillo
    private int filled = 0;

    public SessionDecision(int k, int n) {
        if (n < 1 || k < 1 || k > n) throw new IllegalArgumentException("se requiere 1 <= k <= n");
        this.k = k;
        this.ring = new boolean[n];
    }

    /** Anade el resultado de una ventana; devuelve true si se dispara la alarma. */
    public boolean add(boolean flagged) {
        if (filled == ring.length && ring[pos]) count--;
        ring[pos] = flagged;
        if (flagged) count++;
        pos = (pos + 1) % ring.length;
        filled = Math.min(filled + 1, ring.length);
        if (count >= k) {
            reset();
            return true;
        }
        return false;
    }

    /** Ventanas marcadas actualmente en el historial. */
    public int flaggedInWindow() {
        return count;
    }

    public void reset() {
        java.util.Arrays.fill(ring, false);
        pos = 0;
        count = 0;
        filled = 0;
    }
}
