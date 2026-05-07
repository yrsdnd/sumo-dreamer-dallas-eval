# SUMO + DreamerV3/SafeDreamer — Dallas City Eval Bundle

A self-contained evaluation harness for a **DreamerV3 + SafeDreamer-style PID-Lagrangian** autonomous-driving agent on the **real Dallas city OSM map** in SUMO. Runs the trained model (checkpoint at training step 101,042) on N independent episodes, dumps per-episode rich trajectories, prints aggregate metrics, and renders a cinematic top-down MP4 of the longest successful arrivals.

This bundle is **standalone** — clone the repo, install Python deps + SUMO, download the checkpoint, run one command. No live training. Heavy traffic enforced on every episode (ego is inserted during the peak rush window of the 1-hour Dallas scenario, sim seconds 1500–2400).

The model was trained for ~159k env steps in the parent project [yrsdnd/dreamerv3_torch-main](https://github.com/yrsdnd/dreamerv3_torch-main). Architecture is DreamerV3 (RSSM with `deter=512`, `stoch=32×32` categorical, GRU dynamics, 5×1024 MLP encoder/decoder, 2×512 reward/value heads, 5×1024 cost head, twohot symlog_disc(255) reward+cost critics) with SafeDreamer's PID-Lagrangian safety controller wrapping the actor loss. Full architecture description in the parent repo's `glowing-finding-dahl.md` plan.

## What you get

After running `python run.py`:

```
outputs/
├── eval_summary.json           aggregate metrics + per-episode rows
├── eval_seed000.npz            full trajectory: ego pos/heading/speed,
├── eval_seed001.npz            every visible neighbour vehicle, action,
├── …                           reward, cost — every sim step
└── top3_cinematic.mp4          combined video of the 3 longest arrivals
```

`top3_cinematic.mp4` opens with the full Dallas city visible (ego is a gold dot somewhere in the map), then the camera smoothly zooms in over ~5 seconds onto the ego car and tracks it for the rest of the route. Other vehicles appear as dark-grey rectangles, road network as light-grey lines.

## Quick start (Windows, macOS, or Linux)

### 1. Install Python 3.10 (3.10.x recommended; 3.11 works too)

Windows: download from [python.org](https://www.python.org/downloads/windows/) and run the installer. **Check "Add python.exe to PATH"** during install.

### 2. Install SUMO

Provides the traffic simulator + libsumo Python bindings.

**Windows:** `pip install eclipse-sumo` (in a fresh terminal, after Python is installed). This installs SUMO + libsumo + sumolib in one go and they go on PATH automatically.

**macOS / Linux:** `pip install eclipse-sumo` works the same. Or install via `brew install sumo` (macOS) / `apt install sumo` (Linux) and `pip install libsumo sumolib`.

### 3. Install Python deps

From inside this repo's directory:

```
pip install -r requirements.txt
```

### 4. (Auto) Download the model checkpoint

The first time you run `python run.py`, it auto-downloads the 309 MB checkpoint from this repo's GitHub Release (`v1.0`) into `./checkpoint/`. **Internet required for first run.** Subsequent runs use the cached file.

To download manually:
```
mkdir checkpoint
curl -L -o checkpoint/checkpoint_step_0101042.pt \
  https://github.com/yrsdnd/sumo-dreamer-dallas-eval/releases/download/v1.0/checkpoint_step_0101042.pt
```

### 5. Install ffmpeg (for the MP4 render only)

**Windows:** download a static build from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (release-essentials), unzip, add the `bin/` folder to PATH. Or use `winget install Gyan.FFmpeg`.

**macOS:** `brew install ffmpeg`.
**Linux:** `apt install ffmpeg`.

If you don't want a video, pass `--no-video` to skip this dependency.

### 6. Run

```
python run.py
```

Default: 15 episodes on CPU, heavy-traffic (peak rush hour) window. **~7–15 minutes wall time on a modern laptop CPU.** Output streams to your terminal as each episode finishes.

## Common variants

```bash
# Bigger eval (~50–80 min on CPU, ~30–50 min on GPU)
python run.py --episodes 100

# Faster on Apple Silicon
python run.py --device mps

# Faster on NVIDIA
python run.py --device cuda

# Skip video render (just metrics + trajectories)
python run.py --no-video

# Use a different traffic window (light traffic)
python run.py --heavy-traffic-window 60-300

# Different output folder
python run.py --out-dir my_results
```

## Output format

### `outputs/eval_summary.json`
Top-level dict with:
- `n_episodes`, `success_rate`, `crash_fault_rate`, `wrong_lane_rate`, `mean_reward`, `mean_cost`, `mean_steps`, `mean_sim_seconds`, `mean_wall_seconds`, etc.
- `per_episode`: list of one dict per episode with `seed`, `steps`, `reward`, `cost`, `sim_seconds`, `term_reason`, `outcome`.
- `outcomes`: counter of outcome labels (`arrival`, `collision_fault`, `wrong_lane_end`, `drained`, etc.).

### `outputs/eval_seedNNN.npz`
A NumPy archive per episode. Keys:
- `seed, steps, reward, cost, sim_seconds, wall_seconds, term_reason, outcome`
- `actions` `(T, 2)` — `[accel_norm, lane_signal]` per step
- `rewards`, `costs` `(T,)`
- `ego_x, ego_y` `(T,)` — ego position in UTM Zone 14 metres
- `ego_angle, ego_speed` `(T,)` — SUMO compass deg, m/s
- `nb_x, nb_y, nb_angle, nb_speed, nb_length` — ragged arrays of all visible neighbour vehicles (typically 8-15 cars within 100 m of ego at each step)

## Heavy-traffic enforcement

The `_select_ego_insertion_time()` method on the SUMO env is monkey-patched in `run.py:patch_heavy_traffic()` to always return a time inside the peak-traffic window. Default 1500–2400 sim seconds; configurable via `--heavy-traffic-window MIN-MAX`. The Dallas trip file has 5,896 vehicles distributed over its 3,600-second simulated hour with peak density mid-scenario.

## What's the model doing per step?

Observation (137-dim numeric, all `float32`, all normalised to [-1, 1]):
- `ego` (13): speed, accel, heading sin/cos, lane idx, num lanes, lane pos, speed limit, lane width, road class, ego length, edge-flags
- `neighbors` (10×10): top-10 nearest vehicles in ego frame — presence, longitudinal/lateral offset, relative velocity, heading diff, length, vClass, lane-delta, signal
- `lights` (3×5): up to 3 upcoming TLs — distance, R/Y/G one-hot, time-placeholder
- `route_ctx` (6): dist-to-junction, dist-to-lane-count-change, lane-count delta, dist-to-route-end, heading-offset-from-route, target-lane-delta
- `preferred_lane` (3): is-correct, offset-to-best, fraction-correct

Action (2-dim, continuous):
- `accel_norm ∈ [-1, 1]` → applied as 2.5 m/s² (forward) or -5.0 m/s² (decel)
- `lane_signal ∈ [-1, 1]` → if `|lane_signal| ≥ 0.5`, fires lane-change command (4-second cooldown)

Reward (per step + terminal): progress along route, speed-keeping toward limit, lane-correctness, penalties for futile-merge / oscillation / cross-track-error / heading-misalignment. Terminal: arrival bonus (+25 + 0.025 × route metres) or crash penalty (-50).

Cost (per step + terminal): per-step rule violations (severe speeding, mild speeding in risky context, red-light approach within 3 m), terminal collision-at-fault (+1.0) or victim collision (+0.3).

## Troubleshooting

**"libsumo not found"** → SUMO isn't installed correctly. Try `pip install --upgrade eclipse-sumo` and confirm it appears in `pip list`.

**Video render fails with "ffmpeg: not found"** → Install ffmpeg (see step 5 above) or pass `--no-video`.

**Checkpoint download is slow / times out** → Manually download via the curl command in step 4.

**Eval crashes mid-episode with TraCI error** → SUMO occasionally fails to insert the ego at peak traffic if every nearby lane is occupied. Re-run; the same seed will produce the same outcome (deterministic).

**Episode arrival rate is lower than expected** → Heavy-traffic mode is intentionally harder. The trained agent's success rate on the peak window is around 30-40% over 100 seeds. To compare to lighter traffic, re-run with `--heavy-traffic-window 60-600`.

## Repo layout

```
configs.yaml           # all defaults + 'sumo_dallas_long' preset (used for eval)
run.py                 # main entry — eval + cinematic video
render_video.py        # standalone video renderer (called by run.py)
dreamer.py             # Dreamer agent class
models.py              # WorldModel + ImagBehavior (actor + reward critic + cost critic + slow targets)
networks.py            # RSSM, MultiEncoder, MultiDecoder, MLP, distributions
tools.py               # PIDLagrangian, DiscDist, SymlogDist, Optimizer, …
exploration.py         # Plan2Explore, etc. (unused at eval)
parallel.py            # subprocess env wrapper (unused at eval)
envs/
  ├── __init__.py
  ├── sumo.py          # SumoEnv class (gym.Env)
  ├── wrappers.py      # TimeLimit, SelectAction, UUID
  └── sumo_city_dallas/
       ├── osm.sumocfg
       ├── osm.net.xml.gz       # the Dallas OSM net
       ├── osm.bus.trips.xml
       ├── osm.motorcycle.trips.xml
       ├── osm.passenger.trips.xml
       └── output.add.xml
checkpoint/                # auto-populated on first run
outputs/                   # auto-populated by run.py
CLAUDE_INSTRUCTIONS.md     # explicit step-by-step for Claude Code agents
```

## Provenance

- **Parent project**: [yrsdnd/dreamerv3_torch-main](https://github.com/yrsdnd/dreamerv3_torch-main) (training repo, history of 17+ commits).
- **Architecture**: DreamerV3 (Hafner et al., 2023) base + SafeDreamer (Huang et al., 2023) cost-head + cost-critic + PID-Lagrangian (Stooke et al., 2020).
- **Map**: real Dallas city, OpenStreetMap import, ~6.7 km × 3.9 km, 8,308 edges, 2,009 junctions, 15 traffic lights, in UTM Zone 14.
- **Trip files**: 5,896 vehicles (passenger + motorcycle + bus), 3-5% "dangerous driver" vClass for safety-critical training.

## License

MIT (see `LICENSE`).
