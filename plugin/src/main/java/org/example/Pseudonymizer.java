package org.example;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.UUID;

/**
 * Seudonimizacion con clave: HMAC-SHA256(clave_secreta, UUID).
 *
 * A diferencia de recortar el UUID (que deja 32 bits del identificador real y es reversible buscando
 * el prefijo entre los UUID conocidos), sin la clave no se puede volver al UUID ni enlazar con otros
 * sistemas; con la clave el mismo jugador recibe siempre el mismo seudonimo (trazabilidad forense).
 * Es seudonimizacion, no anonimizacion: quien tenga la clave puede recalcular el seudonimo.
 */
public final class Pseudonymizer {

    public static final int MIN_KEY_BYTES = 16;
    private static final int HEX_CHARS = 16; // 64 bits: colisiones despreciables en la poblacion de un servidor

    private final byte[] key;
    private final ThreadLocal<Mac> mac; // Mac no es thread-safe y la auditoria se escribe en hilos asincronos

    public Pseudonymizer(byte[] key) {
        if (key == null || key.length < MIN_KEY_BYTES) {
            throw new IllegalArgumentException("La clave HMAC debe tener al menos " + MIN_KEY_BYTES + " bytes");
        }
        this.key = key.clone();
        this.mac = ThreadLocal.withInitial(this::newMac);
    }

    /** Clave aleatoria de 32 bytes en Base64 (para guardar en config.yml). */
    public static String generateKey() {
        byte[] k = new byte[32];
        new SecureRandom().nextBytes(k);
        return Base64.getEncoder().encodeToString(k);
    }

    /** Decodifica la clave de config.yml; null si falta o no es valida. */
    public static byte[] decodeKey(String b64) {
        if (b64 == null || b64.isBlank()) return null;
        try {
            byte[] k = Base64.getDecoder().decode(b64.trim());
            return k.length >= MIN_KEY_BYTES ? k : null;
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    public String pseudonym(UUID uuid) {
        return "p-" + hex(hmac(uuid.toString().getBytes(StandardCharsets.UTF_8))).substring(0, HEX_CHARS);
    }

    byte[] hmac(byte[] data) {
        return mac.get().doFinal(data);
    }

    private Mac newMac() {
        try {
            Mac m = Mac.getInstance("HmacSHA256");
            m.init(new SecretKeySpec(key, "HmacSHA256"));
            return m;
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("HmacSHA256 no disponible", e);
        }
    }

    static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder(b.length * 2);
        for (byte x : b) sb.append(Character.forDigit((x >> 4) & 0xF, 16)).append(Character.forDigit(x & 0xF, 16));
        return sb.toString();
    }
}
