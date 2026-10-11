package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FastBreakCheckTest {

    @Test
    void expectedTimeFromBreakSpeed() {
        assertEquals(0.0, FastBreakCheck.expectedMs(1.0f));          // instantaneo
        assertEquals(500.0, FastBreakCheck.expectedMs(0.1f), 1e-9);  // 10 ticks
        assertEquals(350.0, FastBreakCheck.expectedMs(0.15f), 1e-9); // ceil(6.67) = 7 ticks
        assertTrue(Double.isInfinite(FastBreakCheck.expectedMs(0.0f)));
    }

    @Test
    void instantAndCreativeLikeBreaksAreNeverFlagged() {
        assertFalse(FastBreakCheck.tooFast(1.0, 1.0f, 1.0f, 0.7));
        assertFalse(FastBreakCheck.tooFast(1.0, 0.0f, 0.0f, 0.7));   // irrompible: otra regla
    }

    @Test
    void flagsOnlyBelowVanillaTolerance() {
        // bloque de 10 ticks (500 ms): se acepta desde 350 ms
        assertFalse(FastBreakCheck.tooFast(500, 0.1f, 0.1f, 0.7));
        assertFalse(FastBreakCheck.tooFast(350, 0.1f, 0.1f, 0.7));
        assertTrue(FastBreakCheck.tooFast(200, 0.1f, 0.1f, 0.7));
    }

    @Test
    void usesTheMostFavourableSpeed() {
        // picando en el aire (lento) pero rompiendo ya en el suelo (rapido): no se penaliza
        assertFalse(FastBreakCheck.tooFast(400, 0.02f, 0.1f, 0.7));
    }
}
