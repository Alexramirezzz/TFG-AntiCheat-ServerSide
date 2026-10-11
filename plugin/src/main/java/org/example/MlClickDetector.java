package org.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Detector de AutoClicker por ML con ventanas de 50 swings (paso 25).
 *
 * Hilo principal: solo anade el swing al buffer del jugador y, cada 25 swings, copia una ventana.
 * Hilo "CiberPlugin-ML": calcula las features y ejecuta el modelo. La cola es acotada: si se llena
 * la ventana se descarta y se cuenta (backpressure, igual que SessionRecorder), nunca bloquea el tick.
 */
public final class MlClickDetector {

    public static final int WINDOW = 50;
    public static final int STRIDE = 25;
    private static final int QUEUE_CAPACITY = 256;

    @FunctionalInterface
    public interface Scorer {
        float score(float[] x) throws Exception;
    }

    @FunctionalInterface
    public interface Sink {
        /** Llamado desde el hilo de ML; el receptor debe volver al hilo principal si toca Bukkit. */
        void onWindowScored(UUID player, double score, ClickFeatureExtractor.Result features);
    }

    private final Scorer scorer;
    private final Sink sink;
    private final Map<UUID, List<ClickStreamCleaner.Swing>> buffers = new HashMap<>();
    private final ThreadPoolExecutor executor;
    private final AtomicLong dropped = new AtomicLong();
    private final AtomicLong errors = new AtomicLong();

    public MlClickDetector(Scorer scorer, Sink sink) {
        this.scorer = scorer;
        this.sink = sink;
        this.executor = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(QUEUE_CAPACITY),
                r -> {
                    Thread th = new Thread(r, "CiberPlugin-ML");
                    th.setDaemon(true);
                    return th;
                },
                (r, ex) -> dropped.incrementAndGet());
    }

    /** Hilo principal: swing aceptado (no duplicado) del jugador. */
    public void onAccepted(UUID player, ClickStreamCleaner.Swing swing) {
        List<ClickStreamCleaner.Swing> buf = buffers.computeIfAbsent(player, k -> new ArrayList<>(WINDOW + 1));
        buf.add(swing);
        // Se espera al swing 51: asi el contador de duplicados del swing 50 ya es definitivo.
        if (buf.size() < WINDOW + 1) return;
        double[] t = new double[WINDOW];
        boolean[] held = new boolean[WINDOW];
        int[] dup = new int[WINDOW];
        for (int i = 0; i < WINDOW; i++) {
            ClickStreamCleaner.Swing s = buf.get(i);
            t[i] = s.tMs;
            held[i] = s.held;
            dup[i] = s.nDup;
        }
        buf.subList(0, STRIDE).clear();
        executor.execute(() -> score(player, t, held, dup));
    }

    private void score(UUID player, double[] t, boolean[] held, int[] dup) {
        ClickFeatureExtractor.Result f = ClickFeatureExtractor.compute(t, held, dup);
        if (!f.valid || f.routedMining) return; // ventana sin clics suficientes o de minado
        try {
            sink.onWindowScored(player, scorer.score(f.asFloat()), f);
        } catch (Exception e) {
            errors.incrementAndGet();
        }
    }

    public void remove(UUID player) {
        buffers.remove(player);
    }

    public long droppedWindows() {
        return dropped.get();
    }

    public long errors() {
        return errors.get();
    }

    public void shutdown() {
        executor.shutdownNow();
        buffers.clear();
    }
}
