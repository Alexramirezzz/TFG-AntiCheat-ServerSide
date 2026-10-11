package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ClickStreamCleanerLateTickTest {

    @Test
    void lateTickDoesNotEndHeldRun() {
        // secuencia real de la prueba en el servidor: un tick llego a 67 ms y luego 33, 48, 49, 50, 50
        ClickStreamCleaner c = new ClickStreamCleaner();
        c.onDigMarker(0.0);
        assertFalse(c.onSwing(0.4).held);
        double t = 0.4;
        boolean[] expected = {false, true, true, true, true, true};
        double[] iv = {67, 33, 48, 49, 50, 50};
        for (int i = 0; i < iv.length; i++) {
            t += iv[i];
            assertEquals(expected[i], c.onSwing(t).held, "swing " + i);
        }
    }

    @Test
    void clickerAlternatingTicksAfterBlockHitIsNotHeld() {
        ClickStreamCleaner c = new ClickStreamCleaner();
        c.onDigMarker(0.0);
        c.onSwing(0.4);
        double t = 0.4;
        for (int k = 0; k < 20; k++) {
            t += (k % 2 == 0) ? 100.0 : 50.0;
            assertFalse(c.onSwing(t).held, "swing " + k);
        }
    }
}
