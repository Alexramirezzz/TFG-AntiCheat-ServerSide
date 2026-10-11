package org.example;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.scheduler.BukkitTask;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;

public class SessionRecorder {

    private static final Pattern SAFE = Pattern.compile("[A-Za-z0-9_]{1,32}");
    private static final int QUEUE_CAPACITY = 50_000;

    // índices: 0 clicks, 1 moves, 2 breaks, 3 sessions
    private static final String[] NAMES = {"clicks", "moves", "breaks", "sessions"};
    private static final String[] HEADERS = {
            "session_id,t_ns,source,label",
            "session_id,t_ns,dx,dy,dz,on_ground,tps,ping,label",
            "session_id,t_ns,label",
            "session_id,participant,label,start_utc,end_utc,duration_s,events_dropped"
    };

    public static final class Session {
        public final String sessionId;
        public final String participant;
        public final String label;
        final long startNanos = System.nanoTime();
        final Instant startUtc = Instant.now();
        final AtomicLong dropped = new AtomicLong();

        Session(String sessionId, String participant, String label) {
            this.sessionId = sessionId;
            this.participant = participant;
            this.label = label;
        }
    }

    private record Row(int file, String line) {}

    private final Proyecto_ciberPlugin plugin;
    private final Map<UUID, Session> sessions = new ConcurrentHashMap<>();
    private final BlockingQueue<Row> queue = new LinkedBlockingQueue<>(QUEUE_CAPACITY);
    private volatile boolean running;
    private volatile double tps = 20.0;
    private long lastTickCheck = System.nanoTime();
    private Thread writer;
    private BukkitTask tpsTask;

    public SessionRecorder(Proyecto_ciberPlugin plugin) {
        this.plugin = plugin;
    }

    public static boolean isSafe(String s) {
        return SAFE.matcher(s).matches();
    }

    // ---------- ciclo de vida ----------
    public void open() {
        running = true;
        writer = new Thread(this::writeLoop, "CiberPlugin-DatasetWriter");
        writer.setDaemon(true);
        writer.start();

        // TPS medido a mano: una tarea cada 20 ticks deberia tardar 1 s a 20 TPS
        lastTickCheck = System.nanoTime();
        tpsTask = Bukkit.getScheduler().runTaskTimer(plugin, () -> {
            long now = System.nanoTime();
            double elapsedS = (now - lastTickCheck) / 1_000_000_000.0;
            lastTickCheck = now;
            if (elapsedS > 0) tps = Math.min(20.0, 20.0 / elapsedS);
        }, 20L, 20L);
    }

    public void close() {
        for (UUID id : sessions.keySet()) stopSession(id);
        if (tpsTask != null) tpsTask.cancel();
        running = false;
        try {
            if (writer != null) writer.join(3000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ---------- sesiones ----------
    public Session startSession(Player p, String label, String participant) {
        if (!isSafe(label) || !isSafe(participant)) return null;
        String id = UUID.randomUUID().toString().substring(0, 8);
        Session s = new Session(id, participant, label);
        return sessions.putIfAbsent(p.getUniqueId(), s) == null ? s : null;
    }

    public Session stopSession(UUID playerId) {
        Session s = sessions.remove(playerId);
        if (s == null) return null;
        long durationS = (System.nanoTime() - s.startNanos) / 1_000_000_000L;
        enqueue(s, 3, String.join(",", s.sessionId, s.participant, s.label,
                s.startUtc.toString(), Instant.now().toString(),
                Long.toString(durationS), Long.toString(s.dropped.get())));
        return s;
    }

    /** TPS medido por la tarea de este grabador (tambien lo usa el modo conservador de los detectores). */
    public double getTps() {
        return tps;
    }

    public boolean isRecording(UUID playerId) {
        return sessions.containsKey(playerId);
    }

    // ---------- registro de eventos (se llaman desde el hilo principal) ----------
    public void logClick(Player p, String source) {
        Session s = sessions.get(p.getUniqueId());
        if (s == null) return;
        enqueue(s, 0, s.sessionId + "," + (System.nanoTime() - s.startNanos) + "," + source + "," + s.label);
    }

    public void logMove(Player p, double dx, double dy, double dz) {
        Session s = sessions.get(p.getUniqueId());
        if (s == null) return;
        long t = System.nanoTime() - s.startNanos;
        // Locale.ROOT: con locale español %.2f escribiría comas y rompería el CSV
        enqueue(s, 1, String.format(Locale.ROOT, "%s,%d,%.4f,%.4f,%.4f,%d,%.2f,%d,%s",
                s.sessionId, t, dx, dy, dz, p.isOnGround() ? 1 : 0, tps, p.getPing(), s.label));
    }

    public void logBreak(Player p) {
        Session s = sessions.get(p.getUniqueId());
        if (s == null) return;
        enqueue(s, 2, s.sessionId + "," + (System.nanoTime() - s.startNanos) + "," + s.label);
    }

    private void enqueue(Session s, int file, String line) {
        // offer() no bloquea: si la cola está llena se descarta y se cuenta (backpressure)
        if (!queue.offer(new Row(file, line))) s.dropped.incrementAndGet();
    }

    // ---------- hilo escritor ----------
    private void writeLoop() {
        BufferedWriter[] out = new BufferedWriter[NAMES.length];
        try {
            Path dir = plugin.getDataFolder().toPath().resolve("dataset");
            Files.createDirectories(dir);
            for (int i = 0; i < NAMES.length; i++) {
                Path p = dir.resolve(NAMES[i] + ".csv");
                boolean isNew = !Files.exists(p) || Files.size(p) == 0;
                out[i] = Files.newBufferedWriter(p, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                if (isNew) {
                    out[i].write(HEADERS[i]);
                    out[i].newLine();
                }
            }
            int sinceFlush = 0;
            while (running || !queue.isEmpty()) {
                Row r = queue.poll(500, TimeUnit.MILLISECONDS);
                if (r == null) {
                    flushAll(out);
                    sinceFlush = 0;
                    continue;
                }
                out[r.file()].write(r.line());
                out[r.file()].newLine();
                if (++sinceFlush >= 500) {
                    flushAll(out);
                    sinceFlush = 0;
                }
            }
            flushAll(out);
        } catch (IOException e) {
            plugin.getLogger().warning("Error escribiendo dataset: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            for (BufferedWriter w : out) {
                if (w != null) {
                    try { w.close(); } catch (IOException ignored) { }
                }
            }
        }
    }

    private static void flushAll(BufferedWriter[] out) throws IOException {
        for (BufferedWriter w : out) if (w != null) w.flush();
    }
}