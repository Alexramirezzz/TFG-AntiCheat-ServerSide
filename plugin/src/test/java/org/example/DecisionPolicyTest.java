package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DecisionPolicyTest {

    @Test
    void kOfNAlarmAndReset() {
        SessionDecision d = new SessionDecision(3, 8);
        boolean[] seq = {true, false, false, true, false, false, false, false};
        for (boolean b : seq) assertFalse(d.add(b));
        assertEquals(2, d.flaggedInWindow());
        assertFalse(d.add(true));    // the first flag leaves the window: still 2
        assertTrue(d.add(true));     // 3 flags within the last 8 windows
        assertEquals(0, d.flaggedInWindow()); // reset after the alarm
    }

    @Test
    void flagsOutsideTheWindowDoNotAccumulate() {
        SessionDecision d = new SessionDecision(2, 8);
        assertFalse(d.add(true));
        for (int i = 0; i < 8; i++) assertFalse(d.add(false));
        assertFalse(d.add(true));    // previous flag is 9 windows old
        assertTrue(d.add(true));
    }

    @Test
    void lagPolicyBands() {
        LagPolicy p = new LagPolicy(80, 150, 0.25, 18.0);
        assertEquals(1.0, p.alphaFactor(20, 20.0));
        assertEquals(1.0, p.alphaFactor(80, 19.5));
        assertEquals(0.25, p.alphaFactor(120, 20.0));
        assertEquals(LagPolicy.ABSTAIN, p.alphaFactor(200, 20.0));
        assertEquals(LagPolicy.ABSTAIN, p.alphaFactor(10, 15.0));   // servidor lento
        assertTrue(p.abstain(300, 20.0));
        assertFalse(p.abstain(100, 20.0));
    }

    @Test
    void pvalueDefinitionMatchesPython() {
        float[] cal = {1f, 2f, 3f, 4f};
        assertEquals(5 / 5.0, ClickModel.pvalue(cal, 0f), 1e-12);
        assertEquals(3 / 5.0, ClickModel.pvalue(cal, 2.5f), 1e-12);
        assertEquals(2 / 5.0, ClickModel.pvalue(cal, 4f), 1e-12);
        assertEquals(1 / 5.0, ClickModel.pvalue(cal, 10f), 1e-12);
    }
}
