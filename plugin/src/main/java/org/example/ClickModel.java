package org.example;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Ensemble de modelos ONNX descrito en click_model.json (unica clase que depende de ONNX Runtime; si la
 * libreria no esta en el classpath, cargar esta clase lanza NoClassDefFoundError y el plugin vuelve al
 * detector por umbral).
 *
 * Cada modelo devuelve una puntuacion (mayor = mas sospechoso) que se convierte en un p-valor calibrado:
 *   p = (1 + numero de ventanas humanas de calibracion con puntuacion >= s) / (n + 1)
 * es decir, la fraccion de ventanas humanas que parecen igual o mas "automaticas". Una ventana se marca si
 * min(p) < alpha / M (Bonferroni, M = numero de modelos). Es la misma formula que ml/src/decision.py.
 */
public final class ClickModel implements AutoCloseable {

    /** Un modelo del ensemble. */
    private static final class Part {
        final String name;
        final OrtSession session;
        final String inputName;
        final int[] featureIdx;
        final String output;
        final int column;
        final double sign;
        final float[] calSorted;

        Part(String name, OrtSession session, int[] featureIdx, String output, int column, double sign, float[] calSorted) {
            this.name = name;
            this.session = session;
            this.inputName = session.getInputNames().iterator().next();
            this.featureIdx = featureIdx;
            this.output = output;
            this.column = column;
            this.sign = sign;
            this.calSorted = calSorted;
        }
    }

    private final OrtEnvironment env;
    private final List<Part> parts;
    public final double alpha;
    public final int sessionK;
    public final int sessionN;
    public final int lagNormalMaxPing;
    public final int lagStrictMaxPing;
    public final double lagStrictAlphaFactor;
    public final boolean demo;
    public final String description;

    private ClickModel(OrtEnvironment env, List<Part> parts, JsonObject ens, boolean demo) {
        this.env = env;
        this.parts = parts;
        this.alpha = ens.get("alpha").getAsDouble();
        this.sessionK = ens.get("session_k").getAsInt();
        this.sessionN = ens.get("session_n").getAsInt();
        JsonObject lag = ens.getAsJsonObject("lag_policy");
        this.lagNormalMaxPing = lag.get("normal_max_ping").getAsInt();
        this.lagStrictMaxPing = lag.get("strict_max_ping").getAsInt();
        this.lagStrictAlphaFactor = lag.get("strict_alpha_factor").getAsDouble();
        this.demo = demo;
        this.description = ens.get("rule").getAsString() + (demo ? " (DEMO, entrenado con datos sinteticos)" : "");
    }

    /** Carga el ensemble y valida las features contra ClickFeatureExtractor. */
    public static ClickModel load(Path dir, Path metadata) throws IOException, OrtException {
        JsonObject meta = JsonParser.parseString(Files.readString(metadata, StandardCharsets.UTF_8)).getAsJsonObject();
        checkFeatureOrder(meta.getAsJsonArray("features"));
        if (!meta.has("ensemble")) {
            throw new IOException("click_model.json sin seccion 'ensemble' (modelo antiguo)");
        }
        JsonObject ens = meta.getAsJsonObject("ensemble");
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        List<Part> parts = new ArrayList<>();
        try {
            for (JsonElement me : ens.getAsJsonArray("models")) {
                JsonObject m = me.getAsJsonObject();
                JsonArray feats = m.getAsJsonArray("features");
                int[] idx = new int[feats.size()];
                for (int i = 0; i < idx.length; i++) {
                    idx[i] = indexOf(feats.get(i).getAsString());
                }
                JsonArray cal = m.getAsJsonArray("cal_scores");
                float[] calSorted = new float[cal.size()];
                for (int i = 0; i < calSorted.length; i++) calSorted[i] = cal.get(i).getAsFloat();
                Arrays.sort(calSorted);
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                opts.setIntraOpNumThreads(1); // impacto minimo en el servidor
                OrtSession s = env.createSession(dir.resolve(m.get("file").getAsString()).toString(), opts);
                parts.add(new Part(m.get("name").getAsString(), s, idx, m.get("output").getAsString(),
                        m.get("column").getAsInt(), m.get("sign").getAsDouble(), calSorted));
            }
        } catch (IOException | OrtException | RuntimeException e) {
            for (Part p : parts) p.session.close();
            throw e;
        }
        boolean demo = meta.has("demo") && meta.get("demo").getAsBoolean();
        return new ClickModel(env, parts, ens, demo);
    }

    private static void checkFeatureOrder(JsonArray feats) throws IOException {
        if (feats.size() != ClickFeatureExtractor.FEATURES.length) {
            throw new IOException("click_model.json: numero de features distinto al del extractor Java");
        }
        for (int i = 0; i < feats.size(); i++) {
            if (!feats.get(i).getAsString().equals(ClickFeatureExtractor.FEATURES[i])) {
                throw new IOException("click_model.json: feature " + i + " = " + feats.get(i).getAsString()
                        + ", se esperaba " + ClickFeatureExtractor.FEATURES[i]);
            }
        }
    }

    private static int indexOf(String feature) throws IOException {
        for (int i = 0; i < ClickFeatureExtractor.FEATURES.length; i++) {
            if (ClickFeatureExtractor.FEATURES[i].equals(feature)) return i;
        }
        throw new IOException("click_model.json: feature desconocida " + feature);
    }

    public int size() {
        return parts.size();
    }

    public String modelName(int i) {
        return parts.get(i).name;
    }

    /** Puntuacion de cada modelo (mayor = mas sospechoso) para un vector completo de features. */
    public double[] rawScores(float[] x) throws OrtException {
        double[] out = new double[parts.size()];
        for (int k = 0; k < parts.size(); k++) {
            Part p = parts.get(k);
            float[] sub = new float[p.featureIdx.length];
            for (int i = 0; i < sub.length; i++) sub[i] = x[p.featureIdx[i]];
            try (OnnxTensor in = OnnxTensor.createTensor(env, new float[][]{sub});
                 OrtSession.Result res = p.session.run(Collections.singletonMap(p.inputName, in))) {
                Optional<OnnxValue> v = res.get(p.output);
                if (v.isEmpty()) throw new OrtException("Salida " + p.output + " no encontrada en " + p.name);
                Object val = v.get().getValue();
                float s;
                if (val instanceof float[][]) s = ((float[][]) val)[0][p.column];
                else if (val instanceof float[]) s = ((float[]) val)[0];
                else throw new OrtException("Tipo de salida inesperado en " + p.name);
                out[k] = p.sign * s;
            }
        }
        return out;
    }

    /** p-valores calibrados (uno por modelo). */
    public double[] pvalues(float[] x) throws OrtException {
        double[] s = rawScores(x);
        double[] p = new double[s.length];
        for (int k = 0; k < s.length; k++) p[k] = pvalue(parts.get(k).calSorted, (float) s[k]);
        return p;
    }

    /** p = (1 + #{cal >= s}) / (n + 1), con la tabla ordenada (busqueda binaria). */
    static double pvalue(float[] calSorted, float s) {
        int lo = 0, hi = calSorted.length; // primer indice con cal >= s
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (calSorted[mid] < s) lo = mid + 1;
            else hi = mid;
        }
        int ge = calSorted.length - lo;
        return (1.0 + ge) / (calSorted.length + 1.0);
    }

    /** Ventana marcada si min(p) < alpha / M. */
    public boolean flagged(double[] p, double alphaEff) {
        double min = Double.POSITIVE_INFINITY;
        for (double v : p) min = Math.min(min, v);
        return min < alphaEff / p.length;
    }

    @Override
    public void close() throws OrtException {
        for (Part p : parts) p.session.close();
    }
}
