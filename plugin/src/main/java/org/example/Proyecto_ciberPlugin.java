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
import org.bukkit.event.entity.EntityDamageByEntityEvent;
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


    // MAPAS PARA ESTADOS Y SCORING
    // BASTIONADO: Usamos UUID en lugar de Player para evitar Memory Leaks
    private final Map<UUID, Long> lastClickTime = new HashMap<>();
    private final Map<UUID, Long> lastBlockBreakTime = new HashMap<>();
    private final Map<UUID, Integer> violationScores = new HashMap<>();
    private final Map<UUID, Long> lastAlertTime = new HashMap<>();

    // Umbrales en memoria
    private int maxClicksPerSecond;
    private long minBreakDelay;
    private double maxHorizontalDist;
    private double maxCombatReach; 
    private int maxViolations;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        loadConfigValues();

        getServer().getPluginManager().registerEvents(this, this);
        
        if (getCommand("ciberplugin") != null) {
            getCommand("ciberplugin").setExecutor(new ComandosCiberPlugin(this));
        }

        getLogger().info("CiberPlugin ACTIVADO.");
    }

    private void loadConfigValues() {
        int configClicks = getConfig().getInt("deteccion.autoclicker.max_clicks_per_second", 15);
        long configBreak = getConfig().getLong("deteccion.fastbreak.min_delay_ms", 500);
        double configSpeed = getConfig().getDouble("deteccion.speed.max_horizontal_distance", 1.2);
        double configReach = getConfig().getDouble("deteccion.combat.max_reach", 3.8); 
        int configViolations = getConfig().getInt("sanciones.max_violations_before_kick", 10);

        // BASTIONADO: FAIL-SAFE (Validacion de limites)
        this.maxClicksPerSecond = Math.max(5, Math.min(configClicks, 50)); 
        this.minBreakDelay = Math.max(100L, Math.min(configBreak, 2000L)); 
        this.maxHorizontalDist = Math.max(0.5, Math.min(configSpeed, 5.0));
        this.maxCombatReach = Math.max(3.0, Math.min(configReach, 6.0)); 
        this.maxViolations = Math.max(3, Math.min(configViolations, 50));

        if (configClicks != this.maxClicksPerSecond) {
            getLogger().warning("Parametro AutoClicker fuera de limites. Se forzo valor seguro: " + this.maxClicksPerSecond);
        }
    }

    @Override
    public void onDisable() {
        getLogger().info("Proyecto_CiberPlugin DESACTIVADO.");
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        UUID uuid = event.getPlayer().getUniqueId();
        lastClickTime.remove(uuid);
        lastBlockBreakTime.remove(uuid);
        violationScores.remove(uuid);
        lastAlertTime.remove(uuid);
    }


    // MOTOR DE DECISION Y SCORING
    private void flagPlayer(Player player, String cheatType, String evidence) {
        // BASTIONADO: Control de Accesos (RBAC) - Ignorar si tiene permiso
        if (player.hasPermission("ciberplugin.bypass")) return;

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
        Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
            try {
                File logFile = new File(getDataFolder(), "auditoria.log");
                if (!logFile.exists()) {
                    logFile.getParentFile().mkdirs();
                    logFile.createNewFile();
                }
                try (PrintWriter out = new PrintWriter(new FileWriter(logFile, true))) {
                    String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                    
                    // BASTIONADO: Privacidad por Diseno
                    String seudoUUID = player.getUniqueId().toString().substring(0, 8) + "-XXXX-XXXX";

                    out.printf("[%s] ALERT | Actor: %s | Cheat: %s | VL: %d | Evidencia: %s%n", 
                            timestamp, seudoUUID, cheatType, score, evidence);
                }
            } catch (IOException e) {
                getLogger().warning("Error escribiendo auditoria forense: " + e.getMessage());
            }
        });
    }


    // EXPORTACIÓN PARA MACHINE LEARNING
    private void exportTelemetryForML(UUID uuid, long timeBetweenClicks) {
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
                    String seudoUUID = uuid.toString().substring(0, 8);
                    
                    out.printf("%s,%s,%d,unknown%n", timestamp, seudoUUID, timeBetweenClicks);
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

        double horizontalDistance = Math.sqrt(Math.pow(to.getX() - from.getX(), 2) + Math.pow(to.getZ() - from.getZ(), 2));
        double verticalDistance = Math.abs(to.getY() - from.getY());

    
        if (horizontalDistance > maxHorizontalDist) {
            flagPlayer(player, "Speed/Movement", "Distancia H: " + String.format("%.2f", horizontalDistance));
        }


        if (!player.isOnGround() && verticalDistance > 1.0) {
            flagPlayer(player, "Fly", "Distancia V: " + String.format("%.2f", verticalDistance));
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
    @EventHandler
    public void onPlayerInteract(PlayerInteractEvent event) {
        Player player = event.getPlayer();
        UUID uuid = player.getUniqueId();
        long current = System.currentTimeMillis();

        if (lastClickTime.containsKey(uuid)) {
            long last = lastClickTime.get(uuid);
            long diff = current - last;

            exportTelemetryForML(uuid, diff);

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