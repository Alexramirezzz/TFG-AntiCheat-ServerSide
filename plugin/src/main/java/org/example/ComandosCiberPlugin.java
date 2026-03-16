package org.example;

import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;

public class ComandosCiberPlugin implements CommandExecutor {

    private final Proyecto_ciberPlugin plugin;

    public ComandosCiberPlugin(Proyecto_ciberPlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        
        if (args.length > 0 && args[0].equalsIgnoreCase("reload")) {
            
            plugin.reloadPluginConfig();
            
            // -------------------------------------------------
            // BASTIONADO 6: Auditoria de Administradores (Insider Threat)
            // Registra en la consola del servidor quien hizo el cambio
            // -------------------------------------------------
            plugin.getLogger().warning("AUDITORIA SEGURIDAD: El administrador [" + sender.getName() + "] ha recargado la configuracion del Anti-Cheat.");
            
            sender.sendMessage("§8[§a✔§8] §aConfiguracion de seguridad recargada en caliente.");
            return true;
        }
        
        sender.sendMessage("§8[§c!§8] §cUso incorrecto. Usa: /ciberplugin reload");
        return true;
    }
}
