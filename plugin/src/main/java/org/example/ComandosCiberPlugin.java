package org.example;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

public class ComandosCiberPlugin implements CommandExecutor {

    private final Proyecto_ciberPlugin plugin;

    public ComandosCiberPlugin(Proyecto_ciberPlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {

        if (args.length > 0 && args[0].equalsIgnoreCase("reload")) {
            plugin.reloadPluginConfig();

            // BASTIONADO 6: Auditoria de Administradores (Insider Threat)
            // Registra en la consola del servidor quien hizo el cambio
            plugin.getLogger().warning("AUDITORIA SEGURIDAD: El administrador [" + sender.getName() + "] ha recargado la configuracion del Anti-Cheat.");

            sender.sendMessage("§8[§a✔§8] §aConfiguracion de seguridad recargada en caliente.");
            return true;
        }

        // NUEVO: grabacion de sesiones para el dataset
        if (args.length > 0 && args[0].equalsIgnoreCase("record")) {
            return handleRecord(sender, args);
        }

        sender.sendMessage("§8[§c!§8] §cUso incorrecto. Usa: /ciberplugin <reload|record>");
        return true;
    }

    // NUEVO
    private boolean handleRecord(CommandSender sender, String[] args) {
        // BASTIONADO: comprobacion explicita de permisos (defensa en profundidad, ademas del plugin.yml)
        if (!sender.hasPermission("ciberplugin.admin")) {
            sender.sendMessage("§cNo tienes permisos de administrador para usar este comando.");
            return true;
        }

        SessionRecorder rec = plugin.getRecorder();

        // /ciberplugin record start <jugador> <etiqueta> <participante>
        if (args.length >= 5 && args[1].equalsIgnoreCase("start")) {
            Player target = Bukkit.getPlayerExact(args[2]);
            if (target == null) {
                sender.sendMessage("§cJugador no conectado.");
                return true;
            }
            if (rec.isRecording(target.getUniqueId())) {
                sender.sendMessage("§cYa hay una sesión activa para ese jugador.");
                return true;
            }
            SessionRecorder.Session s = rec.startSession(target, args[3], args[4]);
            if (s == null) {
                sender.sendMessage("§cEtiqueta o participante no válidos (solo letras, números y _).");
                return true;
            }
            plugin.getLogger().warning("AUDITORIA SEGURIDAD: [" + sender.getName()
                    + "] inicio grabacion " + s.sessionId + " (" + s.label + ", " + s.participant + ")");
            sender.sendMessage("§aGrabando sesión §e" + s.sessionId + " §a(" + s.label + ", " + s.participant + ")");
            return true;
        }

        // /ciberplugin record stop <jugador>
        if (args.length >= 3 && args[1].equalsIgnoreCase("stop")) {
            Player target = Bukkit.getPlayerExact(args[2]);
            SessionRecorder.Session s = (target == null) ? null : rec.stopSession(target.getUniqueId());
            if (s == null) {
                sender.sendMessage("§cNo había sesión activa para ese jugador.");
            } else {
                plugin.getLogger().warning("AUDITORIA SEGURIDAD: [" + sender.getName()
                        + "] detuvo grabacion " + s.sessionId);
                sender.sendMessage("§aSesión §e" + s.sessionId + " §adetenida.");
            }
            return true;
        }

        sender.sendMessage("§8[§c!§8] §cUso: /ciberplugin record start <jugador> <etiqueta> <participante> | stop <jugador>");
        return true;
    }
}