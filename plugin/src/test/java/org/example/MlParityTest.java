package org.example;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Paridad Java <-> Python. click_features_fixture.json lo genera ml/src/export_onnx.py a partir de
 * las implementaciones de referencia en Python (data.clean_click_stream y features.window_features).
 */
class MlParityTest {

    private static JsonObject fixture() throws Exception {
        try (InputStream in = Objects.requireNonNull(
                MlParityTest.class.getResourceAsStream("/click_features_fixture.json"), "fixture missing")) {
            return JsonParser.parseReader(new InputStreamReader(in, StandardCharsets.UTF_8)).getAsJsonObject();
        }
    }

    @Test
    void featureOrderMatchesPython() throws Exception {
        JsonArray f = fixture().getAsJsonArray("features");
        assertEquals(ClickFeatureExtractor.FEATURES.length, f.size());
        for (int i = 0; i < f.size(); i++) assertEquals(f.get(i).getAsString(), ClickFeatureExtractor.FEATURES[i]);
    }

    @Test
    void cleanerAndFeaturesMatchPython() throws Exception {
        for (JsonElement ce : fixture().getAsJsonArray("cases")) {
            JsonObject c = ce.getAsJsonObject();
            String name = c.get("name").getAsString();
            // merge click events and breaks in time order (same tie rule as Python)
            List<double[]> ev = new ArrayList<>(); // {t, kind} kind: 0 interact_block, 1 interact_air, 2 break, 3 swing
            for (JsonElement e : c.getAsJsonArray("events")) {
                JsonArray a = e.getAsJsonArray();
                String src = a.get(1).getAsString();
                int kind = src.equals("interact_block") ? 0 : src.equals("interact_air") ? 1 : 3;
                ev.add(new double[]{a.get(0).getAsDouble(), kind});
            }
            for (JsonElement b : c.getAsJsonArray("breaks")) ev.add(new double[]{b.getAsDouble(), 2});
            // Python converts to integer ns first; do the same so ties/rounding are identical
            for (double[] e : ev) e[0] = ((long) (e[0] * 1e6)) / 1e6;
            ev.sort((x, y) -> x[0] != y[0] ? Double.compare(x[0], y[0]) : Double.compare(x[1], y[1]));

            ClickStreamCleaner cl = new ClickStreamCleaner();
            List<ClickStreamCleaner.Swing> acc = new ArrayList<>();
            for (double[] e : ev) {
                if (e[1] == 0 || e[1] == 2) cl.onDigMarker(e[0]);
                else if (e[1] == 3) {
                    ClickStreamCleaner.Swing s = cl.onSwing(e[0]);
                    if (s != null) acc.add(s);
                }
            }
            JsonArray expT = c.getAsJsonArray("accepted_t_ms");
            JsonArray expHeld = c.getAsJsonArray("held");
            JsonArray expDup = c.getAsJsonArray("n_dup");
            assertEquals(expT.size(), acc.size(), name + ": accepted swings");
            double[] t = new double[acc.size()];
            boolean[] held = new boolean[acc.size()];
            int[] dup = new int[acc.size()];
            for (int i = 0; i < acc.size(); i++) {
                assertEquals(expT.get(i).getAsDouble(), acc.get(i).tMs, 1e-6, name + " t[" + i + "]");
                assertEquals(expHeld.get(i).getAsBoolean(), acc.get(i).held, name + " held[" + i + "]");
                assertEquals(expDup.get(i).getAsInt(), acc.get(i).nDup, name + " dup[" + i + "]");
                t[i] = acc.get(i).tMs;
                held[i] = acc.get(i).held;
                dup[i] = acc.get(i).nDup;
            }
            ClickFeatureExtractor.Result r = ClickFeatureExtractor.compute(t, held, dup);
            JsonObject ef = c.getAsJsonObject("features");
            assertEquals(ef.get("valid").getAsBoolean(), r.valid, name + " valid");
            assertEquals(ef.get("n_active").getAsInt(), r.nActive, name + " n_active");
            assertEquals(ef.get("held_frac").getAsDouble(), r.heldFrac, 1e-9, name + " held_frac");
            for (String f : ClickFeatureExtractor.FEATURES) {
                JsonElement v = ef.get(f);
                if (v == null || v.isJsonNull()) {
                    assertTrue(Double.isNaN(r.get(f)), name + " " + f + " should be NaN");
                } else {
                    double exp = v.getAsDouble();
                    assertEquals(exp, r.get(f), 1e-6 * Math.max(1.0, Math.abs(exp)), name + " " + f);
                }
            }
        }
    }

    @Test
    void onnxModelMatchesPythonScores() throws Exception {
        Path onnx = Path.of("src/main/resources/models/click_model.onnx");
        Path meta = Path.of("src/main/resources/models/click_model.json");
        assertTrue(Files.exists(onnx), "run ml/src/export_onnx.py first");
        try (ClickModel m = ClickModel.load(onnx, meta)) {
            for (JsonElement se : fixture().getAsJsonArray("model_samples")) {
                JsonObject s = se.getAsJsonObject();
                JsonArray xa = s.getAsJsonArray("x");
                float[] x = new float[xa.size()];
                for (int i = 0; i < x.length; i++) x[i] = xa.get(i).getAsFloat();
                assertEquals(s.get("score").getAsDouble(), m.score(x), 1e-5);
            }
            assertTrue(m.threshold > 0 && m.threshold < 1);
        }
    }

    @Test
    void detectorScoresWindowsOffTheCallerThread() throws Exception {
        List<String> threads = new ArrayList<>();
        java.util.concurrent.CountDownLatch done = new java.util.concurrent.CountDownLatch(1);
        MlClickDetector d = new MlClickDetector(x -> 0.5f, (p, s, f) -> {
            threads.add(Thread.currentThread().getName());
            done.countDown();
        });
        java.util.UUID u = java.util.UUID.randomUUID();
        ClickStreamCleaner c = new ClickStreamCleaner();
        for (int i = 0; i <= MlClickDetector.WINDOW; i++) {
            d.onAccepted(u, c.onSwing(100.0 * i + (i % 3) * 7));
        }
        assertTrue(done.await(5, java.util.concurrent.TimeUnit.SECONDS));
        assertEquals("CiberPlugin-ML", threads.get(0));
        d.shutdown();
    }
}
