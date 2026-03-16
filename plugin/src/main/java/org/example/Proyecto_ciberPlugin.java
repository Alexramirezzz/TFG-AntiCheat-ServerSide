package org.example;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.block.Block;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class Proyecto_ciberPlugin extends JavaPlugin implements Listener {

    // -----------------------------
    // MAPAS PARA ESTADOS Y SCORING
    // -----------------------------
    // BASTIONADO: Usamos UUID en lugar de Player para evitar Memory Leaks
    private final Map<UUID, Long> lastClickTime = new HashMap<>();
    private final Map<UUID, Long> lastBlockBreakTime = new HashMap<>();
    private final Map<UUID, Integer> violationScores = new HashMap<>();
    private final Map<UUID, Long> lastAlertTime = new HashMap<>();

    // Umbrales en memoria
    private int maxClicksPerSecond;
    private long minBreakDelay;
    private double maxHorizontalDist;
    private int maxViolations;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        loadConfigValues();

        getServer().getPluginManager().registerEvents(this, this);
        
        if (getCommand("ciberplugin") != null) {
            getCommand("ciberplugin").setExecutor(new ComandosCiberPlugin(this));
        }

        // Sin tilde para evitar errores de codificacion en Windows
        getLogger().info("Proyecto_CiberPlugin ACTIVADO. Bastionado en ejecucion.");
    }

    private void loadConfigValues() {
        // Leemos valores del archivo
        int configClicks = getConfig().getInt("deteccion.autoclicker.max_clicks_per_second", 15);
        long configBreak = getConfig().getLong("deteccion.fastbreak.min_delay_ms", 500);
        double configSpeed = getConfig().getDouble("deteccion.speed.max_horizontal_distance", 1.2);
        int configViolations = getConfig().getInt("sanciones.max_violations_before_kick", 10);

        // -------------------------------------------------
        // BASTIONADO 1: FAIL-SAFE (Validacion de limites)
        // Evita que un administrador ponga valores absurdos (ej: -5 o 99999) 
        // que puedan causar Denegacion de Servicio (DoS) o inoperatividad.
        // -------------------------------------------------
        this.maxClicksPerSecond = Math.max(5, Math.min(configClicks, 50)); 
        this.minBreakDelay = Math.max(100L, Math.min(configBreak, 2000L)); 
        this.maxHorizontalDist = Math.max(0.5, Math.min(configSpeed, 5.0));
        this.maxViolations = Math.max(3, Math.min(configViolations, 50));

        if (configClicks != this.maxClicksPerSecond) {
            getLogger().warning("Parametro AutoClicker fuera de limites. Se forzo valor seguro: " + this.maxClicksPerSecond);
        }
    }

    @Override
    public void onDisable() {
        getLogger().info("Proyecto_CiberPlugin DESACTIVADO.");
    }

    // BASTIONADO 2: Prevencion de fugas de memoria al desconectarse el jugador
    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        UUID uuid = event.getPlayer().getUniqueId();
        lastClickTime.remove(uuid);
        lastBlockBreakTime.remove(uuid);
        violationScores.remove(uuid);
        lastAlertTime.remove(uuid);
    }

    // -------------------------------------------------
    // MOTOR DE DECISION Y SCORING
    // -------------------------------------------------
    private void flagPlayer(Player player, String cheatType, String evidence) {
        // BASTIONADO 3: Control de Accesos (RBAC) - Ignorar si tiene permiso
        if (player.hasPermission("ciberplugin.bypass")) return;

        UUID uuid = player.getUniqueId();
        int score = violationScores.getOrDefault(uuid, 0) + 1;
        violationScores.put(uuid, score);

        // Control de Spam de alertas (1 segundo de cooldown)
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

        // Registrar evidencia
        logViolationAsync(player, cheatType, evidence, score);

        // Sancion progresiva
        if (score >= maxViolations) {
            Bukkit.getScheduler().runTask(this, () -> {
                player.kickPlayer("§cHas sido desconectado.\n§7Motivo: Comportamiento anomalo detectado.");
                violationScores.remove(uuid);
            });
        }
    }

    private void logViolationAsync(Player player, String cheatType, String evidence, int score) {
        // BASTIONADO 4: Ejecucion Asincrona para proteger los TPS del servidor (Rendimiento)
        Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
            try {
                File logFile = new File(getDataFolder(), "auditoria.log");
                if (!logFile.exists()) {
                    logFile.getParentFile().mkdirs();
                    logFile.createNewFile();
                }
                try (PrintWriter out = new PrintWriter(new FileWriter(logFile, true))) {
                    String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                    
                    // BASTIONADO 5: Privacidad por Diseno (Seudonimizacion del UUID)
                    String seudoUUID = player.getUniqueId().toString().substring(0, 8) + "-XXXX-XXXX";

                    out.printf("[%s] ALERT | Actor: %s | Cheat: %s | VL: %d | Evidencia: %s%n", 
                            timestamp, seudoUUID, cheatType, score, evidence);
                }
            } catch (IOException e) {
                getLogger().warning("Error escribiendo auditoria forense: " + e.getMessage());
            }
        });
    }

    // -------------------------------------------------
    // DETECCIONES (EVENTOS)
    // -------------------------------------------------
    @EventHandler
    public void onPlayerMove(PlayerMoveEvent event) {
        Player player = event.getPlayer();
        Location from = event.getFrom();
        Location to = event.getTo();

        if (to == null) return;

        double horizontalDistance = Math.sqrt(Math.pow(to.getX() - from.getX(), 2) + Math.pow(to.getZ() - from.getZ(), 2));
        double verticalDistance = Math.abs(to.getY() - from.getY());

        if (horizontalDistance > maxHorizontalDist && !player.isFlying()) {
            flagPlayer(player, "Speed/Movement", "Distancia H: " + String.format("%.2f", horizontalDistance));
        }

        if (!player.isFlying() && !player.isOnGround() && verticalDistance > 1.0) {
            flagPlayer(player, "Fly", "Distancia V: " + String.format("%.2f", verticalDistance));
        }

        Block block = to.getBlock();
        if (block.getType().isSolid()) {
            flagPlayer(player, "No-Clip", "Atraveso bloque: " + block.getType().name());
        }
    }

    @EventHandler
    public void onPlayerInteract(PlayerInteractEvent event) {
        Player player = event.getPlayer();
        UUID uuid = player.getUniqueId();
        long current = System.currentTimeMillis();

        if (lastClickTime.containsKey(uuid)) {
            long last = lastClickTime.get(uuid);
            long diff = current - last;

            if (diff > 0 && diff < (1000 / maxClicksPerSecond)) {
                flagPlayer(player, "AutoClicker", "Click delay: " + diff + "ms");
            }
        }
        lastClickTime.put(uuid, current);
    }

    @EventHandler
    public void onBlockBreak(BlockBreakEvent event) {
        Player player = event.getPlayer();
        UUID uuid = player.getUniqueId();
        long current = System.currentTimeMillis();

        if (lastBlockBreakTime.containsKey(uuid)) {
            long last = lastBlockBreakTime.get(uuid);
            long diff = current - last;

            if (diff < minBreakDelay) {
                flagPlayer(player, "FastBreak", "Break delay: " + diff + "ms");
            }
        }
        lastBlockBreakTime.put(uuid, current);
    }
    
    public void reloadPluginConfig() {
        reloadConfig(); 
        loadConfigValues(); 
    }
}
