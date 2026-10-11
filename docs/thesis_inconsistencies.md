# Thesis inconsistencies to fix before showing the TFG

Review of `TFG_Alejandro.pdf` against the code in this repository and the first real recording.
Section and figure numbers refer to the PDF. Numbers marked *real session* come from
`ml/results/data_report.md`: one 65 s session from P01, so they are anecdotal and are no rate
estimate. Items are ordered by how much they could hurt the thesis in front of a reviewer.

## A. Technical claims that the code or data contradict

1. **Speed threshold: 0.1 vs 1.2 vs 0.5.**
   - Annex III (`config.yml`) sets `max_horizontal_distance: 0.1`.
   - Section 4.6.1 says the test exceeded "los 1.2 bloques permitidos por tick". The Annex I code
     default is also 1.2. You mention 1.25 in Figure 3; figures are images, so I could not check
     that value, so please verify it.
   - `loadConfigValues()` clamps the value to `[0.5, 5.0]`, so with the shipped config the
     **effective** threshold is **0.5**, not 0.1 and not 1.2. Only a warning for the AutoClicker
     parameter is printed, so this clamp is silent.
   - *Real session:* 70 of 608 trimmed move events exceed 0.5 blocks/event and 0 exceed 1.2. All
     of them are legitimate (probably creative flight, see item 9).
   - Fix: choose one value, explain it, make text, figure, annex and config agree, and log every
     clamped parameter.

2. **"Ofuscación / seudonimización criptográfica" for a truncated UUID** (sections 3, 4.4.3,
   Figures 16–17, section 5).
   - Keeping the first 8 hex characters and adding `-XXXX-XXXX` is truncation, not cryptography.
     It keeps 32 bits of the real identifier.
   - Minecraft UUIDs are public (Mojang API, server lists), so anyone with a list of candidate
     players can re-identify an actor by prefix match.
   - The branch now uses a keyed **HMAC-SHA256** pseudonym (secret key generated on first run,
     stored in the server's `config.yml`). Update the text, call it *keyed pseudonymisation*
     (GDPR art. 4(5)), and say it is still personal data for whoever holds the key.

3. **The SYN-flood test does not validate the asynchronous design** (section 4.6.3, conclusions).
   - `hping3 --flood` SYN packets are handled by the operating system's TCP stack. They never
     complete a handshake and never reach Minecraft's event loop. They generate no
     `PlayerMoveEvent`/`PlayerInteractEvent` and no `flagPlayer` or async log writes.
   - The TPS drop you observed is network/OS saturation. It says nothing about whether the
     plugin's logging keeps I/O off the main thread. "Certifica empíricamente" is therefore not
     supported.
   - **Valid test proposal:**
     - Run N headless test clients (bot framework, lab server only, N ∈ {1, 10, 50, 100}) that
       join the server and send high-frequency movement and arm-swing/interaction events
       (several per tick), some designed to trigger detectors.
     - Measure with the plugin disabled, with the plugin in `threshold` mode, and in
       `shadow`/`ml` mode.
     - Record server TPS and MSPT (e.g. spark profiler), the recorder's queue size and
       `events_dropped`, the ML engine's dropped-window counter, and the number of async tasks
       created by `logViolationAsync`.
     - Repeat each condition several times and report mean ± CI.
     - This exercises the code paths the thesis claims are protected.
   - Note that `logViolationAsync` opens and closes `auditoria.log` once per flag, in one new
     async task per flag. Under event spam this is not batched; a single writer queue like
     `SessionRecorder`'s would be.

4. **No-Clip uses only the feet block** (section 4.2.2, Annex I `to.getBlock().getType().isSolid()`).
   - The player hitbox is 0.6 × 1.8 blocks and can overlap up to 8 blocks, so the head block and
     neighbouring columns are not checked.
   - `isSolid()` is also true for slabs, stairs, snow layers and similar blocks, which a player
     legitimately stands *inside* (e.g. on a bottom slab the feet are inside the slab's block
     space). That gives false positives.
   - Fix: test the player's bounding box against the blocks' collision shapes (Paper exposes
     `BoundingBox` and block collision shapes), and describe the remaining limitations.

5. **The 15-minute single-player false-positive test is not statistical evidence** (section 4.6.1).
   - One participant, one session, no count of evaluated events, no confidence interval.
     "Zero alerts" in n trials only bounds the false-positive rate by about 3/n at 95% (rule of
     three).
   - The test was run by the developer, who knows the thresholds.
   - *Real session:* replaying the Annex-I rules on the first recorded session gives 1 AutoClicker
     violation (interact events, 33 intervals), 15 FastBreak violations (23 break intervals) and
     7 Fly violations in 65 s of legitimate play. Conditions differ (probably creative mode), but
     this is hard to reconcile with "violationScores siguió a cero".
   - Present it as a smoke test, and run the protocol in `docs/NEXT_STEPS_FOR_ALEJANDRO.md`.

6. **The AutoClicker detector measured the wrong events.**
   - Annex I fires on *every* `PlayerInteractEvent`: right clicks, both hands, and
     `LEFT_CLICK_AIR` + `LEFT_CLICK_BLOCK` from the same physical click.
   - The real data show one physical click producing up to three events within ~5 ms (34 of 110
     raw gaps < 15 ms).
   - Fixed in this branch: swing as the click source, 15 ms de-duplication, held-button
     exclusion. Describe the fix and the reason. Moving to swings *without* the held-button
     exclusion would be worse: 56 of 76 swing intervals in the real session are < 66 ms, because
     holding the button sends one swing per tick.

7. **"Human clicks have jitter, autoclickers do not" is not what the server observes** (sections
   2.3, 4.2.3).
   - The real swing intervals cluster on multiples of 50 ms: 73% of the non-held intervals lie
     within ±5 ms of the grid, against 20% expected without quantization (n = 15, KS p = 0.014).
     This is consistent with the Java client handling mouse input once per 20 Hz client tick.
   - Consequently a fixed-CPS clicker (e.g. 12 CPS = 83 ms) is seen as a mix of 50 and 100 ms
     intervals with CV ≈ 0.3, not as a zero-variance signal.
   - The 66 ms rule catches sub-15-CPS clickers only because quantization creates 50 ms
     intervals, and the same mechanism affects fast-clicking humans.
   - Add this as a limitation of tick-based timing analysis.

8. **Fly and FastBreak rules flag legitimate play.**
   - Fly (`!onGround && |dy| > 1.0`) fires on ordinary falls: vertical speed exceeds 1
     block/tick after a fall of a few blocks. *Real session:* 7 flags, all during a fall.
   - FastBreak (< 450 ms between breaks) fires on instant-break blocks, efficiency tools and
     creative mode. *Real session:* 15 of 23 break intervals.
   - The 3-block "fly" test in section 4.6.1 does not test these legitimate cases.

9. **Game mode is not controlled.** In the real session all 24 breaks arrive ≤ 5 ms after the
   dig start (instant breaks) and there is sustained upward motion of ~0.37 blocks/event. That is
   typical of creative mode, where the speed, fly and fastbreak rules do not apply. State the
   game mode of every test, and consider exempting creative/spectator players from movement
   checks.

10. **Reach check geometry** (section 4.2.4). The distance from the attacker's feet location to
    the victim's feet location is not the eye-to-hitbox distance used by the game. Height
    differences and large mobs therefore distort it, and creative reach is larger. Mention this
    as a limitation, or use eye location to the nearest hitbox point.

11. **Defaults in code differ from `config.yml`.**
    - FastBreak: 500 in the code vs 450 in the config.
    - Max violations: 10 in the code vs 15 in the config.
    - `deteccion.combat.max_reach` is read but is missing from Annex III.
    - Harmless while the config exists, but confusing for auditability. Align them.

12. **Conclusions say the opposite of what is meant** (section 5): "penalizando a los usuarios
    legítimos (falsos positivos) en escenarios propios de la red" should be "*sin* penalizar…".
    The same sentence claims the network uncertainty was managed "de forma efectiva", which no
    experiment measured. Lag was never varied.

## B. References

13. **"Akhunzada, A. A. (2025)" is attributed to the wrong authors.** arXiv:2512.21377, "A
    Systematic Review of Technical Defenses Against Software-Based Cheating in Online
    Multiplayer Games", is by **A. Alangari and O. Alharbi** (submitted 24 Dec 2025). It is cited
    many times in chapter 2; fix every occurrence.
14. **Collins et al. appears twice with different venues.** Both entries carry DOI
    10.1145/3689934.3690816, which is the **CheckMATE '24 workshop** (co-located with CCS '24),
    pp. 30–43. It is not the CCS '24 main proceedings, "pp. 1–15". Merge into one correct entry
    and fix the in-text citations "(Collins, 2024)" vs "(Collins, Poulopoulos, Muench, & Chothia,
    2024)".
15. **Zhang et al. (2024)** (arXiv:2409.14830, HAWK) is now published: per DBLP, IEEE
    Transactions on Information Forensics and Security 21: 240–255 (2026). Cite the journal
    version.
16. **"Cheng, J. W. (2022) … International Journal of Game Theory and Applications"**: I could not
    find this paper or this journal. Verify it or remove it.
17. **"Cadenas. (1986)."** is an incomplete entry with no title. Complete it or remove it.
18. "n.d. de n.d. de 2025" date formats come from the reference manager and should be cleaned
    up. Blogs (Acer, XDA, Miller) are fine as grey literature, but say so.

## C. Structure and editing

19. Two sections are titled "Sistema de Scoring y Gestión del Falso Positivo" (4.3 and 4.4). 4.4
    is the hardening section and should be renamed, e.g. "Bastionado del plugin".
20. Chapter 3 has a "3.2" but no "3.1".
21. The list of figures is shifted by one with respect to the captions (Figure 1 in the text is
    "Figura 2" in the index, and so on).
22. Section 2.11: "sin comprometer la privacidad del cliente haciendo uso de acceso a nivel de
    kernel" says the opposite of what is meant. NoCheatPlus, Vulcan and GrimAC are server-side
    and use *no* kernel access.
23. Section 4.6.1: "después de haber alcanzado los primeros 10 clicks continuos con un delay
    menor a 50 ms" does not match the code (each click under 66 ms is flagged individually) or a
    20 CPS macro (exactly 50 ms).
24. Annex V shows `spigot-api` as `implementation`. It should be `compileOnly`; it works only
    because the jar task does not bundle dependencies. Also, the text says Paper while the build
    uses the Spigot API.
25. Section 3.2 says neural-network training is out of scope and future work suggests Isolation
    Forest. If you include the ML extension, update both, and keep the DEMO/real distinction from
    `docs/ml_report.md`.
