# Protocol: robustness tests under real network lag

Phase F (`ml/src/lag.py`) **simulates** lag by adding per-packet jitter and lag spikes to
recorded or synthetic timestamps. The mapping from ping class to jitter and spike rate
(`PING_CLASSES`) is an assumption. This protocol describes how to replace that assumption with
measurements taken on a real client–server setup.

## Hypotheses from the simulation

* Per-packet jitter spreads held swings beyond the held tolerance (≤ 65 ms, with one late swing
  up to 85 ms tolerated), so some mining swings leak into the "active click" stream and the
  original 66 ms rule flags a legitimate player. On the real session with simulated lag this gives
  1.6 / 6.65 / 9.15 false violations at ping 80 / 150 / 250 ms (`ml/results/phase_f_results.md`).
* Lag spikes release queued packets together, which looks like same-tick double clicks
  (`dup_frac`). Some models' FPR rises with ping (DEMO).

## Setup

1. Paper 1.18.2 test server with the plugin, using `deteccion.autoclicker.engine: shadow` so ML
   scores are logged to `ml_shadow.csv` but never add VL.
2. A client on a second machine, or on the same machine through a loopback shaper.
3. Traffic shaping:
   * **Linux server: `tc netem`** on the egress interface toward the client, e.g.
     `tc qdisc add dev eth0 root netem delay 40ms 8ms distribution normal` (RTT ≈ 80 ms with
     jitter). netem shapes egress only; for symmetric delay, shape both hosts or use an `ifb`
     device for ingress.
   * **Windows: clumsy**, filtering TCP port 25565 in both directions, using "Lag" for delay and
     "Throttle" for bursts. The settings used must be recorded with each session.
4. Conditions: target RTT of 20, 80, 150 and 250 ms, plus one bursty condition (throttling with a
   200–500 ms timeframe).

## Sessions per condition and participant

| block | label | duration |
|---|---|---|
| combat clicking | `human_combat` | 3 × 3 min |
| survival mining | `human_mining` | 2 min |
| reference clicker on the isolated lab server | `auto_fixed` / `auto_jitter` | 2 min each |

Sessions are recorded with `/ciberplugin record start <player> <label> <participant>`, in
survival mode, with the network condition encoded in the participant code (e.g. `P03_ping80`).

## Measurements

The recorder already stores the client-reported `ping` and a TPS estimate in `moves.csv`.

1. **Shaping check:** per-session median and IQR of `ping`.
2. **Jitter actually received:** the client sends movement every tick, so the spread of
   `moves.csv` inter-arrival times around 50 ms estimates the per-packet jitter seen by the
   server. The fitted values per condition replace `PING_CLASSES` in `ml/src/lag.py`.
3. **Held-button robustness:** fraction of swings classified as held in `human_mining` sessions,
   per condition. If it drops with ping, the held tolerance should become adaptive (e.g. 65 ms
   plus three times the measured jitter sd), in both `config.py` and `ClickStreamCleaner`.
4. **Detection under lag:** re-running `ml/run_all.ps1` reports FPR on real human windows,
   broken down per condition through the participant code.
5. **Server impact:** TPS with the ML engine off and in `shadow` mode, at each ping.

Real-lag results are reported separately from the simulated ones, with the number of
participants, sessions and windows per condition.
