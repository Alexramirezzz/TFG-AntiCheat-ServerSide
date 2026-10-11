# TODO - real network-lag tests (replace the simulated Phase F)

`ml/src/lag.py` only **simulates** lag: it adds per-packet jitter and lag spikes to recorded or
synthetic timestamps. The mapping from ping class to jitter/spike rate (`PING_CLASSES`) is an
assumption. These steps replace the assumption with measurements. Only you can run them, because
they need the real client, server and network.

## What the simulation suggested (to be confirmed or refuted)

* Per-packet jitter of 8–25 ms is enough to break the held-button rule (held swings are expected
  every 50 ± 15 ms). Mining swings then leak into the "active click" stream, and the original
  66 ms rule starts flagging a legitimate player (real timestamps + simulated lag, see
  `ml/results/phase_f_results.md`).
* Lag spikes release queued packets together, which looks like same-tick double clicks
  (`dup_frac`). Some models' FPR rises with ping (DEMO).

## Setup

1. Paper 1.18.2 test server (`C:\paper-server`) with the branch jar; set
   `deteccion.autoclicker.engine: shadow` so ML scores are logged but never add VL.
2. Client on a second machine, or the same machine through a loopback shaper.
3. Shaping tool:
   * **Linux server (preferred): `tc netem`** on the server's egress interface toward the client,
     e.g. `sudo tc qdisc add dev eth0 root netem delay 40ms 8ms distribution normal`
     (RTT ≈ 80 ms ± jitter). Remove it with `sudo tc qdisc del dev eth0 root`.
     netem only shapes **egress**, so for symmetric delay either shape both hosts or use an
     `ifb` device for ingress.
   * **Windows: clumsy** (`clumsy.exe`, filter
     `outbound and tcp.DstPort == 25565 or inbound and tcp.SrcPort == 25565`), with "Lag" for
     delay and "Throttle" for bursts. Note that clumsy's "Lag" is a constant delay. Add jitter by
     combining it with "Throttle" or "Out of order" off, and record the settings you used.
4. Conditions: target RTT of 20 / 80 / 150 / 250 ms, plus at least one "spiky" condition
   (Throttle with a 200–500 ms timeframe).

## Protocol per condition

For each participant (with consent) and each condition:

| block | label | duration |
|---|---|---|
| combat clicking | `human_combat` | 3 × 3 min |
| mining (survival mode) | `human_mining` | 2 min |
| your own clicker test, local machine only, labelled | `auto_fixed` / `auto_jitter` | 2 min each |

Use `/ciberplugin record start <player> <label> <participant>`. Add the condition to the
participant code, e.g. `P03_ping80`, so that `sessions.csv` keeps it. Better still, add a
`condition` column to the recorder later. **Play in survival mode.** The first real session
looks like creative mode (24/24 instant breaks).

## What to measure

The recorder already stores `ping` and `tps` in `moves.csv` (the client-reported ping and a TPS
estimate measured every 20 ticks).

1. Check that the shaping worked: per-session median and IQR of `ping` in `moves.csv`.
2. Measure the jitter that actually arrives: inter-arrival times of `moves.csv` rows (the client
   sends movement every tick) minus 50 ms. Their spread is the real per-packet jitter at the
   server. Fit it per condition and replace `PING_CLASSES` in `ml/src/lag.py` with the
   measured values.
3. Held-button robustness: fraction of swings classified as held in `human_mining` sessions per
   condition (from `data/processed/real_click_streams.csv`). If it drops sharply with ping,
   widen `HELD_MAX_IV_MS` adaptively (e.g. 65 ms + 3 × the measured jitter sd) in **both** Python
   (`config.py`) and Java (`ClickStreamCleaner`).
4. Re-run `ml\run_all.ps1`. The REAL section of `phase_d_results.md` then reports FPR on real
   human windows, now per condition if you encode it in the participant code.
5. Compare TPS with and without the ML engine at each ping (`tps` column), which feeds the
   performance claim.

## Reporting

Report real-lag results separately from the simulated ones, with the number of participants,
sessions and windows per condition. Keep the simulated table, labelled "simulated", for
comparison.
