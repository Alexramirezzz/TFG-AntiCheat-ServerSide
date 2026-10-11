# Next steps that only you can do

Everything in `ml/` runs, but every detection metric so far is a **DEMO on synthetic humans**.
The real data (one 65 s session from P01) supports only the data report, the per-event behaviour
of the original rules, and the timing-quantization observation. The steps below turn the
pipeline into evidence. After each data collection, run:

```powershell
.\ml\run_all.ps1            # from the main checkout, where ml\data\raw\ lives
```

When real windows of both classes exist, the REAL section of
`ml/results/phase_d_results.md` fills in automatically (grouped by participant when at least
two participants per class exist; otherwise grouped by session, with a printed limitation).

## 1. Record real sessions (with consent)

**Ethics and consent (do this first).**
- Written informed consent per participant: purpose, which data are recorded (timestamps of
  clicks, breaks, movement deltas, ping and TPS; no chat, no names), retention period, and the
  right to withdraw and have sessions deleted.
- Minors need guardian consent; ask your tutor whether the university needs an ethics-committee
  approval.
- Use participant codes (`P02`, `P03`, …) only. Keep the code↔person mapping offline, outside
  the repository.
- `ml/data/raw/` is git-ignored; keep it that way.

**Protocol.**
- Survival mode, same server and plugin version.
- Record each block with `/ciberplugin record start <player> <label> <participant>` and
  `/ciberplugin record stop <player>`.
- Note the game mode, the input device and the ping in a local sheet.
- Write down if a session crashed: sessions without a `sessions.csv` row are kept but flagged.

| label | what the participant does | per participant | why |
|---|---|---|---|
| `human_combat` | fight mobs / a dummy, normal clicking; include some fast "jitter/butterfly" clicking if that is their style | 3 × 5 min | main negative class; needs fast human clickers |
| `human_mining` | normal survival mining (held button) | 2 × 3 min | checks the held-button routing |
| `human_explore` | walk, sprint, jump, fall, swim | 1 × 5 min | movement FPR (speed/fly) |
| `auto_fixed` | your own clicker at 10, 13 and 16 CPS, **lab server only** | 3 × 2 min | positive class: fixed rate |
| `auto_jitter` | same clicker with its randomisation on | 3 × 2 min | positive class: jittered |

**Targets.** At ~8 CPS of active clicking, 15 min of combat gives about 150–200 fifty-click
windows per participant.
- **≥ 10 participants** gives ~2,000 human combat windows. That allows 5-fold grouped
  cross-validation by participant and a meaningful 1% FPR operating point (~400–500
  calibration windows per fold).
- **FPR ≤ 0.1%** needs several thousand calibration windows, i.e. **≥ 30 participants**.
  Below that, report 0.1% as "below the measurement resolution".
- **Clickers:** run them on at least 3 different machines/clicker programs if possible, so
  that "participant" grouping for attacks is not one single clicker.

Once you have ≥ 300 non-held human intervals, `synth_humans.fit_hyper` automatically switches
the synthetic humans from the placeholder to parameters fitted to your data.

## 2. Real lag tests

Follow `docs/TODO_real_lag_tests.md` (tc netem on Linux, or clumsy on Windows, at 20 / 80 / 150 /
250 ms plus a spiky condition). Run the plugin in `engine: shadow` so ML scores are logged to
`ml_shadow.csv` without adding VL. The key questions are:
- whether the held-button rule survives real jitter;
- what per-packet jitter actually reaches the server (inter-arrival spread of `moves.csv`).

## 3. Bot-based stress test (replaces the SYN-flood argument)

See item 3 of `docs/thesis_inconsistencies.md`.
- Lab server only.
- N headless test clients (1/10/50/100) joining and sending high-frequency movement and swings.
- Measure TPS/MSPT (spark), the recorder's `events_dropped`, the ML engine's dropped windows
  (printed on disable/reload), and CPU, with the plugin off, in `threshold` mode and in
  `shadow` mode.
- Repeat each condition at least 3 times and report mean ± CI.

## 4. Test the jar (I did not run the server)

1. Build the jar with `.\gradlew.bat build -x test`; it is
   `plugin\build\libs\mi-anti-cheat-plugin.jar`. Copy it to `C:\paper-server\plugins\`.
2. First start:
   - the server must download `onnxruntime` (Spigot library loader, needs internet);
   - `plugins/<plugin>/config.yml` gets a generated `privacidad.hmac_key`. Do not share or commit
     it.
3. Check that `auditoria.log` now shows `Actor: p-<16 hex>`.
4. Default `engine: threshold`. Verify that holding the button to mine does **not** raise
   AutoClicker VL, and that fast air-clicking above 15 CPS still does.
5. Set `engine: shadow`, then `/ciberplugin reload`. The console should show the model and the
   DEMO warning, and `ml_shadow.csv` should fill as you click.
6. Set an invalid engine name, then reload. It must fall back to `threshold` with a warning
   (fail-safe).
7. Do **not** use `engine: ml` for sanctions until the model is retrained and evaluated on real
   data.

## 5. Re-run and update the thesis

After new data:
1. Run `.\ml\run_all.ps1`.
2. Re-read `ml/results/data_report.md` and the `phase_*_results.md` files.
3. Update `docs/ml_report.md`: the numbers in it were produced by the current run, so replace
   them, and move results from "DEMO" to "real" only when they come from real windows.

Then fix the items in `docs/thesis_inconsistencies.md`.
