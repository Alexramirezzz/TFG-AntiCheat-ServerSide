package org.example;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

/**
 * Envoltorio del modelo ONNX (unica clase que depende de ONNX Runtime). Si la libreria no esta en el
 * classpath, cargar esta clase lanza NoClassDefFoundError y el plugin vuelve al detector por umbral.
 */
public final class ClickModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    public final double threshold;
    public final boolean demo;
    public final String description;

    private ClickModel(OrtEnvironment env, OrtSession session, double threshold, boolean demo, String description) {
        this.env = env;
        this.session = session;
        this.inputName = session.getInputNames().iterator().next();
        this.threshold = threshold;
        this.demo = demo;
        this.description = description;
    }

    /** Carga el modelo y valida que su lista de features coincide con ClickFeatureExtractor. */
    public static ClickModel load(Path onnx, Path metadata) throws IOException, OrtException {
        JsonObject meta = JsonParser.parseString(Files.readString(metadata, StandardCharsets.UTF_8)).getAsJsonObject();
        JsonArray feats = meta.getAsJsonArray("features");
        if (feats.size() != ClickFeatureExtractor.FEATURES.length) {
            throw new IOException("click_model.json: numero de features distinto al del extractor Java");
        }
        for (int i = 0; i < feats.size(); i++) {
            if (!feats.get(i).getAsString().equals(ClickFeatureExtractor.FEATURES[i])) {
                throw new IOException("click_model.json: feature " + i + " = " + feats.get(i).getAsString()
                        + ", se esperaba " + ClickFeatureExtractor.FEATURES[i]);
            }
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(1); // impacto minimo en el servidor
        OrtSession session = env.createSession(onnx.toString(), opts);
        double thr = meta.get("threshold_fpr_1pct").getAsDouble();
        boolean demo = meta.has("demo") && meta.get("demo").getAsBoolean();
        String desc = meta.get("model").getAsString() + (demo ? " (DEMO, entrenado con datos sinteticos)" : "");
        return new ClickModel(env, session, thr, demo, desc);
    }

    /** Probabilidad de automatizacion para un vector de features (orden FEATURES). */
    public float score(float[] x) throws OrtException {
        try (OnnxTensor in = OnnxTensor.createTensor(env, new float[][]{x});
             OrtSession.Result out = session.run(Collections.singletonMap(inputName, in))) {
            for (Map.Entry<String, OnnxValue> e : out) {
                Object v = e.getValue().getValue();
                if (v instanceof float[][]) {
                    float[][] p = (float[][]) v;
                    return p[0][p[0].length - 1];
                }
            }
            throw new OrtException("El modelo no devolvio probabilidades");
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
