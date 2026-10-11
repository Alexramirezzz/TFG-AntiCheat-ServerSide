package org.example;

import java.util.Arrays;

/**
 * Caracteristicas de una ventana de clics (port exacto de ml/src/features.py window_features).
 * Codigo puro (sin Bukkit): se ejecuta en el hilo de ML, nunca en el hilo principal.
 */
public final class ClickFeatureExtractor {

    /** Orden de entrada del modelo ONNX (debe coincidir con click_model.json). */
    public static final String[] FEATURES = {"mean", "std", "cv", "skew", "kurt", "min", "p05", "p95",
            "entropy", "ac1", "max_cps_1s", "tick_frac", "dup_frac", "pause_frac"};

    public static final double PAUSE_MS = 1000.0;
    public static final int MIN_INTERVALS = 10;
    public static final double HELD_ROUTE_FRAC = 0.5;
    public static final double TICK_MS = 50.0;
    public static final double TICK_NEAR_MS = 5.0;
    private static final double HIST_BIN_MS = 25.0;
    private static final int HIST_BINS = 21; // 20 bins de 25 ms en [0, 500) + desbordamiento

    private ClickFeatureExtractor() { }

    public static final class Result {
        public final double[] values = new double[FEATURES.length];
        public boolean valid;
        public boolean routedMining;
        public double heldFrac;
        public int nActive;
        public int nSwings;

        public float[] asFloat() {
            float[] x = new float[values.length];
            for (int i = 0; i < x.length; i++) x[i] = (float) values[i];
            return x;
        }

        public double get(String name) {
            for (int i = 0; i < FEATURES.length; i++) if (FEATURES[i].equals(name)) return values[i];
            throw new IllegalArgumentException(name);
        }
    }

    /**
     * @param t     instantes de los swings aceptados (ms), ordenados
     * @param held  swing generado por boton mantenido
     * @param nDup  duplicados absorbidos por cada swing
     */
    public static Result compute(double[] t, boolean[] held, int[] nDup) {
        Result r = new Result();
        int n = t.length;
        r.nSwings = n;
        Arrays.fill(r.values, Double.NaN);
        if (n == 0) return r;

        int nHeld = 0;
        long dupSum = 0;
        for (int i = 0; i < n; i++) {
            if (held[i]) nHeld++;
            dupSum += nDup[i];
        }
        r.heldFrac = (double) nHeld / n;
        r.routedMining = r.heldFrac >= HELD_ROUTE_FRAC;

        // intervalos conservados: el swing i no es "held"; pausas (> 1 s) fuera de las estadisticas
        double[] a = new double[n];
        int kept = 0, pauses = 0, na = 0;
        for (int i = 1; i < n; i++) {
            if (held[i]) continue;
            double d = t[i] - t[i - 1];
            kept++;
            if (d > PAUSE_MS) pauses++;
            else a[na++] = d;
        }
        a = Arrays.copyOf(a, na);
        r.nActive = na;
        r.valid = na >= MIN_INTERVALS;
        set(r, "dup_frac", (double) dupSum / (n + dupSum));
        set(r, "pause_frac", kept > 0 ? (double) pauses / kept : Double.NaN);
        if (!r.valid) return r;

        double mean = 0;
        for (double v : a) mean += v;
        mean /= na;
        double m2 = 0, m3 = 0, m4 = 0;
        for (double v : a) {
            double d = v - mean, d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        double den = m2;            // suma de cuadrados (para la autocorrelacion)
        m2 /= na; m3 /= na; m4 /= na;
        double std = Math.sqrt(m2);
        set(r, "mean", mean);
        set(r, "std", std);
        set(r, "cv", mean > 0 ? std / mean : 0.0);
        set(r, "skew", m2 > 1e-12 ? m3 / Math.pow(m2, 1.5) : 0.0);
        set(r, "kurt", m2 > 1e-12 ? m4 / (m2 * m2) - 3.0 : 0.0);

        double[] s = a.clone();
        Arrays.sort(s);
        set(r, "min", s[0]);
        set(r, "p05", percentileLinear(s, 0.05));
        set(r, "p95", percentileLinear(s, 0.95));

        int[] counts = new int[HIST_BINS];
        for (double v : a) counts[Math.min((int) Math.floor(v / HIST_BIN_MS), HIST_BINS - 1)]++;
        double h = 0;
        for (int c : counts) {
            if (c == 0) continue;
            double p = (double) c / na;
            h -= p * (Math.log(p) / Math.log(2));
        }
        set(r, "entropy", h);

        double num = 0;
        for (int i = 0; i < na - 1; i++) num += (a[i] - mean) * (a[i + 1] - mean);
        set(r, "ac1", den > 1e-12 && na > 1 ? num / den : 0.0);

        int near = 0;
        for (double v : a) {
            double res = v - TICK_MS * Math.rint(v / TICK_MS); // rint = redondeo al par (como numpy)
            if (Math.abs(res) <= TICK_NEAR_MS) near++;
        }
        set(r, "tick_frac", (double) near / na);
        set(r, "max_cps_1s", maxClicksIn1s(t, held));
        return r;
    }

    static double percentileLinear(double[] sorted, double q) {
        int n = sorted.length;
        double pos = q * (n - 1);
        int lo = (int) Math.floor(pos);
        int hi = Math.min(lo + 1, n - 1);
        return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
    }

    static int maxClicksIn1s(double[] t, boolean[] held) {
        double[] act = new double[t.length];
        int m = 0;
        for (int i = 0; i < t.length; i++) if (!held[i]) act[m++] = t[i];
        int best = 0, j = 0;
        for (int i = 0; i < m; i++) {
            while (act[i] - act[j] >= 1000.0) j++;
            best = Math.max(best, i - j + 1);
        }
        return best;
    }

    private static void set(Result r, String name, double v) {
        for (int i = 0; i < FEATURES.length; i++) {
            if (FEATURES[i].equals(name)) {
                r.values[i] = v;
                return;
            }
        }
        throw new IllegalArgumentException(name);
    }
}
