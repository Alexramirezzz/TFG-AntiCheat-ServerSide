package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ClickStreamCleanerTest {

    @Test
    void duplicateSwingsAreMergedIntoLastAccepted() {
        ClickStreamCleaner c = new ClickStreamCleaner();
        assertNotNull(c.onSwing(0));
        assertNull(c.onSwing(10));          // < 15 ms -> duplicate
        ClickStreamCleaner.Swing s = c.onSwing(20);
        assertNotNull(s);
        assertEquals(20.0, s.ivMs, 1e-9);
        assertNull(c.onSwing(31));
        assertEquals(1, s.nDup);
    }

    @Test
    void instantBreakThenHeldSwingsAreHeld() {
        // pattern seen in the real data: interact_block, break, the click's swing, then one swing per tick
        ClickStreamCleaner c = new ClickStreamCleaner();
        c.onDigMarker(0.0);
        c.onDigMarker(0.2);
        assertFalse(c.onSwing(0.6).held);   // the press itself is a real click
        assertTrue(c.onSwing(50.1).held);
        assertTrue(c.onSwing(105.4).held);
        c.onDigMarker(657.0);
        assertFalse(c.onSwing(657.6).held); // new press after release
        assertTrue(c.onSwing(705.0).held);
    }

    @Test
    void fastClicksInAirAreNeverHeld() {
        ClickStreamCleaner c = new ClickStreamCleaner();
        for (int k = 0; k < 20; k++) {
            ClickStreamCleaner.Swing s = c.onSwing(50.0 * k);
            assertFalse(s.held);
        }
    }

    @Test
    void releaseEndsHeldRun() {
        ClickStreamCleaner c = new ClickStreamCleaner();
        c.onDigMarker(1000);
        c.onSwing(1002);
        for (int k = 1; k < 7; k++) assertTrue(c.onSwing(1002 + 50.0 * k).held);
        assertFalse(c.onSwing(1802).held);  // 500 ms later: released and clicked again
        assertFalse(c.onSwing(1850).held);  // no dig marker -> not held
    }
}
