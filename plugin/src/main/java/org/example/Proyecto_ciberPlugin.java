package org.example;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.block.Block;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerAnimationEvent;
import org.bukkit.event.player.PlayerAnimationType;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.EquipmentSlot;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

public class Proyecto_ciberPlugin extends JavaPlugin implements Listener {

    // MAPAS PARA ESTADOS Y SCORING
    // BASTIONADO: Usamos UUID en lugar de Player para evitar Memory Leaks
    // Limpieza del flujo de clics (swing + de-duplicacion + boton mantenido), una por jugador
    private final Map<UUID, ClickStreamCleaner> clickCleaners = new HashMap<>();
    // FastBreak: inicio de la excavacion actual (bloque, instante y velocidad de rotura)
    private final Map<UUID, DigStart> digStarts = new HashMap<>();
    private final Map<UUID, Integer> violationScores = new HashMap<>();
    private final Map<UUID, Long> lastAlertTime = new HashMap<>();

    // Umbrales en memoria
    private int maxClicksPerSecond;
    private double fastBreakTolerance;
    private double maxHorizontalDist;
    private double maxCombatReach;
    private int maxViolations;

    // NUEVO: grabador de sesiones para el dataset
    private SessionRecorder recorder;

    // NUEVO: motor de AutoClicker. threshold = detector original (por defecto); shadow = umbral activo y el
    // modelo ML solo se registra (sin VL); ml = el modelo ML suma VL y el umbral queda como respaldo.
    private volatile String clickEngine = "threshold";   // leido tambien desde el hilo de ML
    private double mlThresholdOverride;
    private boolean mlCustomModel;        // true = respetar el modelo de la carpeta del plugin
    private MlClickDetector mlDetector;   // null si el motor ML no esta activo o no pudo cargarse
    private volatile ClickModel clickModel;
    private volatile double mlThreshold;

    // BASTIONADO: seudonimizacion con clave (HMAC-SHA256) para auditoria y telemetria
    private volatile Pseudonymizer pseudonymizer;

    // NUEVO
    public SessionRecorder getRecorder() {
        return recorder;
    }

    @Override
    public void onEnable() {
        saveDefaultConfig();
        loadConfigValues();
        loadPseudonymizer();

        recorder = new SessionRecorder(this); // NUEVO
        recorder.open();                      // NUEVO
        initMlEngine();                       // NUEVO

        getServer().getPluginManager().registerEvents(this, this);

        if (getCommand("ciberplugin") != null) {
            getCommand("ciberplugin").setExecutor(new ComandosCiberPlugin(this));
        }

        getLogger().info("CiberPlugin ACTIVADO.");
    }

    private void loadConfigValues() {
        int configClicks = getConfig().getInt("deteccion.autoclicker.max_clicks_per_second", 15);
        double configBreakTol = getConfig().getDouble("deteccion.fastbreak.tolerance", 0.7);
        double configSpeed = getConfig().getDouble("deteccion.speed.max_horizontal_distance", 1.2);
        double configReach = getConfig().getDouble("deteccion.combat.max_reach", 3.8);
        int configViolations = getConfig().getInt("sanciones.max_violations_before_kick", 10);

        // BASTIONADO: FAIL-SAFE (Validacion de limites)
        this.maxClicksPerSecond = Math.max(5, Math.min(configClicks, 50));
        this.fastBreakTolerance = Math.max(0.3, Math.min(configBreakTol, 1.0));
        this.maxHorizontalDist = Math.max(0.5, Math.min(configSpeed, 5.0));
        this.maxCombatReach = Math.max(3.0, Math.min(configReach, 6.0));
        this.maxViolations = Math.max(3, Math.min(configViolations, 50));

        if (configClicks != this.maxClicksPerSecond) {
            getLogger().warning("Parametro AutoClicker fuera de limites. Se forzo valor seguro: " + this.maxClicksPerSecond);
        }

        // BASTIONADO: FAIL-SAFE del selector de motor (valor desconocido -> detector por umbral)
        String engine = getConfig().getString("deteccion.autoclicker.engine", "threshold").trim().toLowerCase(Locale.ROOT);
        if (!engine.equals("threshold") && !engine.equals("shadow") && !engine.equals("ml")) {
            getLogger().warning("deteccion.autoclicker.engine invalido (" + engine + "). Se usa 'threshold'.");
            engine = "threshold";
        }
        this.clickEngine = engine;
        double thr = getConfig().getDouble("deteccion.autoclicker.ml_threshold", 0.0);
        // 0 = usar el umbral calibrado del modelo; si se fija a mano se limita a [0.5, 0.9999]
        this.mlThresholdOverride = thr <= 0 ? 0 : Math.max(0.5, Math.min(thr, 0.9999));
        this.mlCustomModel = getConfig().getBoolean("deteccion.autoclicker.ml_custom_model", false);
    }

    // BASTIONADO: clave HMAC en config.yml (privacidad.hmac_key). Si falta o es invalida se genera una
    // nueva con SecureRandom y se guarda; nunca se escribe en los logs.
    private void loadPseudonymizer() {
        byte[] key = Pseudonymizer.decodeKey(getConfig().getString("privacidad.hmac_key", ""));
        if (key == null) {
            String fresh = Pseudonymizer.generateKey();
            getConfig().set("privacidad.hmac_key", fresh);
            saveConfig();
            key = Pseudonymizer.decodeKey(fresh);
            getLogger().warning("AUDITORIA SEGURIDAD: generada una nueva clave de seudonimizacion (privacidad.hmac_key). "
                    + "Los seudonimos anteriores no se pueden enlazar con los nuevos. No compartas esta clave.");
        }
        this.pseudonymizer = new Pseudonymizer(key);
    }

    // MOTOR ML: carga el modelo ONNX y, ante cualquier fallo, vuelve al detector por umbral (fail-safe)
    private void initMlEngine() {
        shutdownMlEngine();
        if (clickEngine.equals("threshold")) return;
        try {
            File dir = new File(getDataFolder(), "models");
            // Por defecto se usa siempre el modelo incluido en el jar (asi un jar nuevo trae su modelo nuevo).
            // Con ml_custom_model: true se respetan los ficheros que el administrador haya puesto en models/.
            boolean useBundled = !mlCustomModel || !new File(dir, "click_model.onnx").exists()
                    || !new File(dir, "click_model.json").exists();
            if (useBundled) {
                saveResource("models/click_model.onnx", true);
                saveResource("models/click_model.json", true);
            }
            clickModel = ClickModel.load(new File(dir, "click_model.onnx").toPath(),
                    new File(dir, "click_model.json").toPath());
            mlThreshold = mlThresholdOverride > 0 ? mlThresholdOverride : clickModel.threshold;
            final ClickModel model = clickModel;
            mlDetector = new MlClickDetector(model::score, this::onMlWindowScored);
            getLogger().info("Motor AutoClicker '" + clickEngine + "' con modelo " + model.description
                    + (useBundled ? " incluido en el plugin" : " personalizado (models/)")
                    + ", umbral " + String.format(Locale.ROOT, "%.4f", mlThreshold));
            if (model.demo) {
                getLogger().warning("El modelo ML es una DEMO entrenada con datos sinteticos: usar 'shadow' hasta validarlo con datos reales.");
            }
        } catch (Throwable t) { // incluye NoClassDefFoundError si ONNX Runtime no esta disponible
            getLogger().warning("Motor ML no disponible (" + t + "). Se usa el detector por umbral.");
            shutdownMlEngine();
            clickEngine = "threshold";
        }
    }

    private void shutdownMlEngine() {
        if (mlDetector != null) {
            if (mlDetector.droppedWindows() > 0 || mlDetector.errors() > 0) {
                getLogger().warning("Motor ML: ventanas descartadas=" + mlDetector.droppedWindows()
                        + ", errores=" + mlDetector.errors());
            }
            mlDetector.shutdown();
            mlDetector = null;
        }
        if (clickModel != null) {
            try { clickModel.close(); } catch (Exception ignored) { }
            clickModel = null;
        }
    }

    // Llamado desde el hilo de ML: toda accion sobre Bukkit vuelve al hilo principal
    private void onMlWindowScored(UUID uuid, double score, ClickFeatureExtractor.Result f) {
        if (!isEnabled()) return;
        boolean flagged = score > mlThreshold;
        String evidence = String.format(Locale.ROOT,
                "score=%.3f umbral=%.3f | entropia=%.2f bits, media=%.0f ms, cv=%.2f, ac1=%.2f, max_cps=%.0f%s",
                score, mlThreshold, f.get("entropy"), f.get("mean"), f.get("cv"), f.get("ac1"),
                f.get("max_cps_1s"), clickModel != null && clickModel.demo ? " (modelo DEMO)" : "");
        if (clickEngine.equals("shadow")) {
            logMlShadowAsync(uuid, score, flagged, evidence);
            return;
        }
        if (!flagged) return;
        Bukkit.getScheduler().runTask(this, () -> {
            Player p = Bukkit.getPlayer(uuid);
            if (p != null && p.isOnline()) flagPlayer(p, "AutoClicker-ML", evidence);
        });
    }

    @Override
    public void onDisable() {
        shutdownMlEngine();                     // NUEVO
        if (recorder != null) recorder.close(); // NUEVO
        getLogger().info("Proyecto_CiberPlugin DESACTIVADO.");
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        UUID uuid = event.getPlayer().getUniqueId();
        recorder.stopSession(uuid); // NUEVO
        clickCleaners.remove(uuid);
        if (mlDetector != null) mlDetector.remove(uuid);
        digStarts.remove(uuid);
        violationScores.remove(uuid);
        lastAlertTime.remove(uuid);
    }

    // MOTOR DE DECISION Y SCORING
    private void flagPlayer(Player player, String cheatType, String evidence) {
        // BASTIONADO: Control de Accesos (RBAC) - Ignorar si tiene permiso
        if (player.hasPermission("ciberplugin.bypass")) return;

        // NUEVO: no sancionar mientras se graba una sesion
        if (recorder.isRecording(player.getUniqueId())) return;

        UUID uuid = player.getUniqueId();
        int score = violationScores.getOrDefault(uuid, 0) + 1;
        violationScores.put(uuid, score);

        // Control de Spam de alertas
        long now = System.currentTimeMillis();
        if (now - lastAlertTime.getOrDefault(uuid, 0L) > 1000) {
            lastAlertTime.put(uuid, now);

            String alertMsg = String.format("§8[§c!§8] §c%s §7fallo §e%s §8[VL: %d] §7(%s)",
                    player.getName(), cheatType, score, evidence);

            for (Player p : Bukkit.getOnlinePlayers()) {
                if (p.hasPermission("ciberplugin.admin")) {
                    p.sendMessage(alertMsg);
                }
            }
        }

        logViolationAsync(player, cheatType, evidence, score);

        if (score >= maxViolations) {
            Bukkit.getScheduler().runTask(this, () -> {
                player.kickPlayer("§cHas sido desconectado.\n§7Motivo: Comportamiento anomalo detectado.");
                violationScores.remove(uuid);
            });
        }
    }

    private void logViolationAsync(Player player, String cheatType, String evidence, int score) {
        // Seudonimo calculado en el hilo principal (no se accede al objeto Player desde el hilo asincrono)
        final String actor = pseudonym(player.getUniqueId());
        Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
            try {
                File logFile = new File(getDataFolder(), "auditoria.log");
                if (!logFile.exists()) {
                    logFile.getParentFile().mkdirs();
                    logFile.createNewFile();
                }
                try (PrintWriter out = new PrintWriter(new FileWriter(logFile, true))) {
                    String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

                    // BASTIONADO: Privacidad por Diseno (seudonimo HMAC-SHA256 con clave, no UUID recortado)
                    out.printf("[%s] ALERT | Actor: %s | Cheat: %s | VL: %d | Evidencia: %s%n",
                            timestamp, actor, cheatType, score, evidence);
                }
            } catch (IOException e) {
                getLogger().warning("Error escribiendo auditoria forense: " + e.getMessage());
            }
        });
    }

    // MODO SHADOW: se registra cada ventana puntuada por el modelo sin sumar VL (validacion en produccion)
    private void logMlShadowAsync(UUID uuid, double score, boolean flagged, String evidence) {
        String actor = pseudonym(uuid);
        Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
            try {
                File f = new File(getDataFolder(), "ml_shadow.csv");
                boolean isNew = !f.exists();
                if (isNew) {
                    f.getParentFile().mkdirs();
                    f.createNewFile();
                }
                try (PrintWriter out = new PrintWriter(new FileWriter(f, true))) {
                    if (isNew) out.println("timestamp,actor,score,threshold,would_flag,evidence");
                    String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"));
                    out.printf(Locale.ROOT, "%s,%s,%.4f,%.4f,%b,\"%s\"%n", timestamp, actor,
                            score, mlThreshold, flagged, evidence.replace("\"", "'"));
                }
            } catch (IOException e) {
                getLogger().warning("Error escribiendo ml_shadow.csv: " + e.getMessage());
            }
        });
    }

    // Seudonimo del jugador para los registros (HMAC-SHA256 con la clave secreta del servidor)
    private String pseudonym(UUID uuid) {
        return pseudonymizer.pseudonym(uuid);
    }

    // EXPORTACIÓN PARA MACHINE LEARNING
    private void exportTelemetryForML(UUID uuid, long timeBetweenClicks) {
        final String actor = pseudonym(uuid);
        Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
            try {
                File mlFile = new File(getDataFolder(), "telemetria_ml.csv");
                boolean isNew = !mlFile.exists();
                if (isNew) {
                    mlFile.getParentFile().mkdirs();
                    mlFile.createNewFile();
                }
                try (PrintWriter out = new PrintWriter(new FileWriter(mlFile, true))) {
                    if (isNew) out.println("timestamp,uuid_ofuscado,delay_ms,label");

                    String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"));
                    out.printf("%s,%s,%d,unknown%n", timestamp, actor, timeBetweenClicks);
                }
            } catch (IOException e) {
            }
        });
    }

    // DETECCIONES DE MOVIMIENTO (SPEED / FLY)
    @EventHandler
    public void onPlayerMove(PlayerMoveEvent event) {
        Player player = event.getPlayer();
        Location from = event.getFrom();
        Location to = event.getTo();

        if (to == null) return;

        // NUEVO: registrar movimiento (ignora giros de camara)
        double dx = to.getX() - from.getX();
        double dy = to.getY() - from.getY();
        double dz = to.getZ() - from.getZ();
        if (dx != 0 || dy != 0 || dz != 0) {
            recorder.logMove(player, dx, dy, dz);
        }

        // CORREGIDO: en creativo/espectador, volando, con elitros, con tridente (riptide) o en un vehiculo el
        // movimiento es legitimo aunque supere los limites de un jugador a pie.
        if (isMovementExempt(player)) return;

        double horizontalDistance = Math.sqrt(Math.pow(to.getX() - from.getX(), 2) + Math.pow(to.getZ() - from.getZ(), 2));

        if (horizontalDistance > maxHorizontalDist) {
            flagPlayer(player, "Speed/Movement", "Distancia H: " + String.format("%.2f", horizontalDistance));
        }

        // CORREGIDO: antes se usaba |dy| y una caida normal (mas de ~5 bloques) superaba 1 bloque/tick.
        // Solo se marca un ascenso anomalo: un salto normal sube 0.42 bloques/tick.
        if (!player.isOnGround() && dy > 1.0) {
            flagPlayer(player, "Fly", "Ascenso V: " + String.format("%.2f", dy));
        }

        Block block = to.getBlock();
        if (block.getType().isSolid()) {
            flagPlayer(player, "No-Clip", "Atraveso bloque: " + block.getType().name());
        }
    }

    // NUEVA DETECCIÓN DE COMBATE (REACH / KILLAURA)
    @EventHandler
    public void onPlayerAttack(EntityDamageByEntityEvent event) {
        if (event.getDamager() instanceof Player) {
            Player attacker = (Player) event.getDamager();
            Location attackerLoc = attacker.getLocation();
            Location victimLoc = event.getEntity().getLocation();

            double distance = attackerLoc.distance(victimLoc);

            if (distance > maxCombatReach) {
                flagPlayer(attacker, "Reach/KillAura", "Distancia ataque: " + String.format("%.2f", distance) + " bloques");
            }
        }
    }

    // DETECCIONES DE AUTOMATIZACIÓN (CLICS)
    // CORREGIDO: un clic fisico genera hasta 3 eventos en ~5 ms (LEFT_CLICK_BLOCK, LEFT_CLICK_AIR y el
    // swing), y medir intervalos entre eventos de interaccion daba falsos positivos. Ahora:
    //  - el clic es el swing del brazo (onSwing), de-duplicado a 15 ms;
    //  - LEFT_CLICK_BLOCK y las roturas solo marcan una excavacion, para no evaluar los swings que el
    //    cliente genera cada tick mientras se mantiene pulsado el boton (minar).
    @EventHandler
    public void onPlayerInteract(PlayerInteractEvent event) {
        Action action = event.getAction();
        if (action != Action.LEFT_CLICK_AIR && action != Action.LEFT_CLICK_BLOCK) return;
        if (event.getHand() != EquipmentSlot.HAND) return;

        Player player = event.getPlayer();
        recorder.logClick(player, action == Action.LEFT_CLICK_AIR ? "interact_air" : "interact_block");

        if (action == Action.LEFT_CLICK_BLOCK) {
            cleanerOf(player.getUniqueId()).onDigMarker(nowMs());
            Block clicked = event.getClickedBlock();
            if (clicked != null) {
                digStarts.put(player.getUniqueId(), new DigStart(clicked, nowMs(), clicked.getBreakSpeed(player)));
            }
        }
    }

    // Inicio de una excavacion: bloque, instante (ms) y velocidad de rotura en ese momento
    private record DigStart(Block block, double tMs, float speed) { }

    private static boolean isMovementExempt(Player p) {
        GameMode gm = p.getGameMode();
        return gm == GameMode.CREATIVE || gm == GameMode.SPECTATOR || p.getAllowFlight() || p.isFlying()
                || p.isGliding() || p.isRiptiding() || p.isInsideVehicle();
    }

    // El swing del brazo es la fuente del clic
    @EventHandler
    public void onSwing(PlayerAnimationEvent event) {
        if (event.getAnimationType() != PlayerAnimationType.ARM_SWING) return;
        Player player = event.getPlayer();
        recorder.logClick(player, "swing");

        UUID uuid = player.getUniqueId();
        ClickStreamCleaner.Swing swing = cleanerOf(uuid).onSwing(nowMs());
        if (swing == null) return;                                   // duplicado
        if (mlDetector != null) mlDetector.onAccepted(uuid, swing);  // el modelo ve tambien los swings "held"
        if (swing.held || Double.isNaN(swing.ivMs)) return;          // minando / primer clic
        if (clickEngine.equals("ml")) return;                        // en modo ml el umbral queda de respaldo

        long diff = (long) Math.floor(swing.ivMs);
        exportTelemetryForML(uuid, diff);

        if (diff > 0 && diff < (1000 / maxClicksPerSecond)) {
            flagPlayer(player, "AutoClicker", "Click delay: " + diff + "ms");
        }
    }

    private ClickStreamCleaner cleanerOf(UUID uuid) {
        return clickCleaners.computeIfAbsent(uuid, k -> new ClickStreamCleaner());
    }

    private static double nowMs() {
        return System.nanoTime() / 1_000_000.0;
    }

    @EventHandler
    public void onBlockBreak(BlockBreakEvent event) {
        Player player = event.getPlayer();
        UUID uuid = player.getUniqueId();
        double now = nowMs();

        recorder.logBreak(player); // NUEVO
        cleanerOf(uuid).onDigMarker(now); // la rotura tambien abre/continua una excavacion

        // CORREGIDO: antes se comparaba el tiempo entre roturas con 450 ms fijos, lo que marcaba bloques que
        // se rompen al instante, herramientas con eficiencia y el modo creativo. Ahora se compara el tiempo
        // de excavacion de ESTE bloque con el que corresponde a su dureza, la herramienta y los efectos.
        DigStart start = digStarts.remove(uuid);
        if (player.getGameMode() == GameMode.CREATIVE || player.getGameMode() == GameMode.SPECTATOR) return;
        Block block = event.getBlock();
        if (start == null || !start.block().equals(block)) return; // sin inicio conocido: no hay evidencia
        double elapsed = now - start.tMs();
        float speedEnd = block.getBreakSpeed(player);
        if (FastBreakCheck.tooFast(elapsed, start.speed(), speedEnd, fastBreakTolerance)) {
            double expected = FastBreakCheck.expectedMs(Math.max(start.speed(), speedEnd));
            flagPlayer(player, "FastBreak", String.format(Locale.ROOT, "Rotura en %.0f ms (esperado %.0f ms, %s)",
                    elapsed, expected, block.getType().name()));
        }
    }

    public void reloadPluginConfig() {
        reloadConfig();
        loadConfigValues();
        loadPseudonymizer();
        initMlEngine();
    }
}