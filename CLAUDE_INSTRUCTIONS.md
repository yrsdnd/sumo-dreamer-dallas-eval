# Instructions for a Claude Code agent on Windows

These are the **exact** steps a fresh Claude Code chat should follow to clone this repo, install dependencies, run the evaluation, and produce results + video on a Windows desktop. Read this top to bottom before executing.

## 0. Prerequisites the user must have

- Windows 10 or 11 with PowerShell or Windows Terminal.
- Python 3.10 or 3.11 installed and on PATH (`python --version` should print `Python 3.10.x` or `3.11.x`).
- Git for Windows installed (`git --version` works).
- Roughly 2 GB of free disk (309 MB checkpoint + Python deps + outputs).
- Internet (to clone, to download the checkpoint, to fetch Python wheels).

If any of those is missing, ask the user to install it before continuing.

## 1. Clone the repo

```powershell
cd $env:USERPROFILE\Desktop
git clone https://github.com/yrsdnd/sumo-dreamer-dallas-eval.git
cd sumo-dreamer-dallas-eval
```

Verify you see `run.py`, `dreamer.py`, `envs/sumo_city_dallas/osm.net.xml.gz`, and the rest of the layout listed in `README.md`.

## 2. Install dependencies

Create a fresh virtual environment (recommended — keeps deps isolated):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `Activate.ps1` is blocked by execution policy, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
then retry.

**Verify SUMO is importable:**
```powershell
python -c "import libsumo; print('libsumo OK'); import sumolib; print('sumolib OK')"
```
Both must print "OK". If either fails:
- `pip install --upgrade eclipse-sumo` (this single package usually provides everything)
- Restart the terminal so newly-installed binaries appear on PATH.

**Verify torch loads:**
```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

## 3. Install ffmpeg (for the MP4 render)

If `ffmpeg -version` already prints info, skip this step.

Otherwise, the simplest install on Windows:
```powershell
winget install -e --id Gyan.FFmpeg
```

If `winget` is unavailable, download a static build from `https://www.gyan.dev/ffmpeg/builds/` (the "release-essentials" 7z), extract to e.g. `C:\ffmpeg\`, and add `C:\ffmpeg\bin` to PATH. Restart the terminal.

If the user explicitly doesn't want a video, you can skip ffmpeg and pass `--no-video` to `run.py`.

## 4. Run the evaluation

For a quick first run (~7-15 min on CPU, 15 episodes):
```powershell
python run.py
```

For the full eval (~50-80 min on CPU, 100 episodes, more reliable stats):
```powershell
python run.py --episodes 100
```

If the user has an NVIDIA GPU and wants speed:
```powershell
python run.py --device cuda
```

The first run will auto-download the 309 MB checkpoint from this repo's GitHub Release. Subsequent runs use the cached file at `checkpoint/checkpoint_step_0101042.pt`.

The script prints a header showing checkpoint path + heavy-traffic window, then one line per episode (`steps / return / cost / outcome`), then a final aggregate block. **Do not interrupt** until you see the final `DONE.` line — Ctrl-C mid-eval will abort and leave incomplete results.

## 5. Verify outputs

After the run completes, the user should have:

```
outputs/
├── eval_summary.json
├── eval_seed000.npz   (and one per seed)
├── …
└── top3_cinematic.mp4   ← the cinematic video of the 3 longest arrivals
```

Verify all three exist and that `top3_cinematic.mp4` is at least 500 KB (a successful render produces 1–3 MB per episode, ~3–9 MB combined).

Open the video for the user automatically:
```powershell
Start-Process .\outputs\top3_cinematic.mp4
```

This launches the default video player (usually Films & TV).

If `top3_cinematic.mp4` is missing (e.g. ffmpeg failed), the trajectories and summary are still valid — re-run with `--no-video` skipped after fixing ffmpeg, or render manually:
```powershell
python render_video.py --eval-dir outputs --out outputs\top3_cinematic.mp4
```

## 6. Show the user the headline metrics

Open `outputs\eval_summary.json` and read out:
- `n_episodes`
- `success_rate` × 100 → "X% arrival rate on heavy-traffic Dallas"
- `crash_fault_rate` × 100 → "Y% collision-at-fault"
- `mean_reward`, `mean_cost`, `mean_steps`

Optionally show the per-episode breakdown:
```powershell
python -c "
import json
with open('outputs/eval_summary.json') as f:
    s = json.load(f)
print(f'Episodes: {s[\"n_episodes\"]}')
print(f'Success rate: {s[\"success_rate\"]*100:.1f}%')
print(f'Crash-fault: {s[\"crash_fault_rate\"]*100:.1f}%')
print(f'Wrong-lane: {s[\"wrong_lane_rate\"]*100:.1f}%')
print(f'Mean reward: {s[\"mean_reward\"]:+.2f}')
print(f'Mean cost: {s[\"mean_cost\"]:.2f}')
print(f'Per-episode:')
for ep in s['per_episode']:
    print(f'  seed={ep[\"seed\"]:>3} steps={ep[\"steps\"]:>5} ret={ep[\"reward\"]:+8.2f} '
          f'cost={ep[\"cost\"]:6.1f} outcome={ep[\"outcome\"]}')
"
```

## Common errors and fixes

**`ModuleNotFoundError: No module named 'libsumo'`** — `pip install eclipse-sumo` (the bundled libsumo is what we use). Restart the terminal after install.

**`FileNotFoundError ... osm.net.xml.gz`** — you're not running `run.py` from the repo root. `cd` to the directory containing `run.py`.

**`RuntimeError: Could not load checkpoint ...`** — checkpoint download was incomplete. Delete `checkpoint/checkpoint_step_0101042.pt` and re-run; `run.py` will re-fetch it.

**Eval is taking forever** — the model is running on CPU. Use `--device cuda` if NVIDIA, `--device mps` if Apple Silicon. Or reduce `--episodes 5` for a smoke test.

**`ffmpeg: command not found`** — install ffmpeg (see step 3) or pass `--no-video`.

**Eval ran but `top3_cinematic.mp4` says "no arrival episodes"** — none of the episodes ended in `arrival`. Increase `--episodes` (e.g. 30+) so at least one route succeeds, or widen the traffic window: `--heavy-traffic-window 60-2400`.

## Where to ask the user before deciding

- If `--episodes` is unclear, ask: 15 episodes (~10 min) for a quick demo, or 100 (~50-80 min) for project-quality stats?
- If the user has an NVIDIA GPU, ask: use `--device cuda`?
- If something errors that isn't covered here, paste the full traceback to the user before guessing fixes.

## Don't

- Don't commit or push to this repo from the user's machine without explicit permission.
- Don't change `configs.yaml`, `dreamer.py`, `models.py`, etc. — they have to match the trained checkpoint exactly. The eval script monkey-patches just one method (`_select_ego_insertion_time`) for heavy-traffic and that's it.
- Don't disable the heavy-traffic patch unless the user asks.
- Don't move `checkpoint/checkpoint_step_0101042.pt` outside the repo unless `--checkpoint <path>` is also passed to `run.py`.
