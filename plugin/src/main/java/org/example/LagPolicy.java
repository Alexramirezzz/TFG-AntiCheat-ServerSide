package org.example;

/**
 * Modo conservador segun la calidad de la conexion y del servidor (ml/src/decision.py, LAG_POLICY).
 *
 * El jitter de red desplaza los intervalos entre clics y aumenta los falsos positivos de los detectores de
 * tiempo (fase F). Con ping normal se evalua igual; con ping alto se exige mas evidencia (alpha menor); con
 * ping muy alto o TPS bajo no se evalua (abstencion), porque la senal deja de ser fiable.
 * Riesgo conocido: un tramposo podria simular lag para no ser evaluado; por eso las abstenciones se registran.
 */
public final class LagPolicy {

    /** Multiplicador de alpha que indica abstencion. */
    public static final double ABSTAIN = 0.0;

    private final int normalMaxPing;
    private final int strictMaxPing;
    private final double strictFactor;
    private final double minTps;

    public LagPolicy(int normalMaxPing, int strictMaxPing, double strictFactor, double minTps) {
        this.normalMaxPing = normalMaxPing;
        this.strictMaxPing = Math.max(strictMaxPing, normalMaxPing);
        this.strictFactor = strictFactor;
        this.minTps = minTps;
    }

    /** Factor por el que se multiplica alpha: 1 (normal), strictFactor (estricto) o ABSTAIN. */
    public double alphaFactor(int pingMs, double tps) {
        if (tps < minTps) return ABSTAIN;
        if (pingMs <= normalMaxPing) return 1.0;
        if (pingMs <= strictMaxPing) return strictFactor;
        return ABSTAIN;
    }

    /** Para los detectores por umbral: true si la conexion o el servidor no permiten evaluar con fiabilidad. */
    public boolean abstain(int pingMs, double tps) {
        return alphaFactor(pingMs, tps) == ABSTAIN;
    }
}
