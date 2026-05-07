"""Single-command evaluation runner for the SUMO + DreamerV3/SafeDreamer
Dallas-city autonomous-driving model (checkpoint at step 101,042).

Run:
    python run.py                 # 15 episodes on CPU, heavy traffic
    python run.py --episodes 100  # full eval
    python run.py --device cuda   # NVIDIA GPU on Windows
    python run.py --device mps    # Apple Silicon GPU

Outputs go to ./outputs/:
    outputs/eval_summary.json     # aggregate metrics
    outputs/eval_seedNNN.npz      # per-episode rich trajectories
    outputs/run.log               # stdout capture
    outputs/top3_cinematic.mp4    # cinematic video of 3 longest arrivals

The script enforces HEAVY TRAFFIC throughout every eval episode by
overriding the env's curriculum to always insert the ego in the peak
traffic window (sim seconds 1500-2400 — the busiest stretch of the
1-hour Dallas scenario). No empty-road episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------- bootstrap
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SUMO_DBG", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Heavy-traffic ego-insertion window (SUMO sim seconds).
# Dallas trip file spans 0-3600s; density peaks ~1500-2400s.
HEAVY_TRAFFIC_TMIN = 1500.0
HEAVY_TRAFFIC_TMAX = 2400.0

CHECKPOINT_REL_PATH = "checkpoint/checkpoint_step_0101042.pt"
CHECKPOINT_DOWNLOAD_URL = (
    "https://github.com/yrsdnd/sumo-dreamer-dallas-eval/releases/"
    "download/v1.0/checkpoint_step_0101042.pt"
)
CHECKPOINT_SIZE_MB_APPROX = 309  # for the user prompt


def ensure_checkpoint(repo_root: Path) -> Path:
    """Download the model checkpoint from the GitHub Release if missing."""
    target = repo_root / CHECKPOINT_REL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    print(f"Downloading checkpoint (~{CHECKPOINT_SIZE_MB_APPROX} MB) …")
    print(f"  from: {CHECKPOINT_DOWNLOAD_URL}")
    print(f"  to:   {target}")
    tmp = target.with_suffix(target.suffix + ".part")
    urlretrieve(CHECKPOINT_DOWNLOAD_URL, str(tmp))
    tmp.rename(target)
    if target.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Downloaded file is only {target.stat().st_size} bytes — "
            f"the URL is probably wrong. Delete {target} and check "
            f"https://github.com/yrsdnd/sumo-dreamer-dallas-eval/releases."
        )
    print(f"  done ({target.stat().st_size // (1024*1024)} MB).")
    return target


def load_config(preset_name: str, device: str, logdir: Path):
    import ruamel.yaml as yaml
    cfg_path = HERE / "configs.yaml"
    _yaml = yaml.YAML(typ="safe", pure=True)
    raw = _yaml.load(cfg_path.read_text())

    def recurse(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                recurse(base[k], v)
            else:
                base[k] = v

    merged = {}
    for name in ["defaults", preset_name]:
        recurse(merged, raw[name])
    ns = Namespace(**merged)
    ns.device = device
    ns.compile = False
    ns.precision = 32
    ns.logdir = str(logdir)
    ns.traindir = str(logdir / "train_eps")
    ns.evaldir = str(logdir / "eval_eps")
    ns.envs = 1
    ns.parallel = False
    return ns


def patch_heavy_traffic(base_env, tmin: float, tmax: float):
    """Override the env's t_insert curriculum to always pick from the
    peak-traffic window. Each eval episode now starts mid-rush-hour."""
    rng = base_env._rng

    def _heavy_traffic_t_insert():
        return rng.uniform(tmin, tmax)

    base_env._select_ego_insertion_time = _heavy_traffic_t_insert


def build_agent_and_env(args, ckpt_path, logdir):
    import numpy as np
    import torch
    import dreamer as dreamer_mod
    import envs.sumo as sumo_env
    import envs.wrappers as wrappers

    config = load_config("sumo_dallas_long", args.device, logdir)

    env = sumo_env.SumoEnv(task="dallas_city", seed=args.seeds[0])
    base_env = env  # raw SumoEnv before wrappers
    env = wrappers.TimeLimit(env, int(config.time_limit))
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    class Logger:
        step = 0
        def __init__(self): pass
        def scalar(self, n, v): pass
        def image(self, *a, **k): pass
        def video(self, *a, **k): pass
        def write(self, *a, **k): pass

    class EmptyDataset:
        def __next__(self): raise StopIteration

    agent = dreamer_mod.Dreamer(
        env.observation_space, env.action_space, config, Logger(), EmptyDataset()
    ).to(config.device)
    agent.requires_grad_(False)
    agent.eval()
    ckpt = torch.load(str(ckpt_path), map_location=config.device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    if agent._pid is not None and "pid_state_dict" in ckpt:
        agent._pid.load_state_dict(ckpt["pid_state_dict"])
    print(f"  loaded checkpoint step {ckpt.get('step', '?')}")
    if agent._pid is not None:
        print(f"  pid state: λ_p={agent._pid.lagrange_penalty:.3f} (frozen at eval)")

    return agent, env, base_env, config


def classify_outcome(term_reason: str) -> str:
    if not term_reason:
        return "other"
    tr = term_reason.lower()
    if "arriv" in tr:
        return "arrival"
    if "collision" in tr or "crash" in tr:
        return "collision_victim" if "victim" in tr else "collision_fault"
    if "wrong_lane" in tr:
        return "wrong_lane_end"
    if "stuck" in tr:
        return "stuck"
    if "silent" in tr:
        return "silent_stop"
    if "drain" in tr or "scenario" in tr:
        return "drained"
    if "timeout" in tr or "time_limit" in tr:
        return "timeout"
    return f"other:{term_reason}"


def capture_state(libsumo_traci, libsumo_constants, EGO_ID):
    out = {"ego_x": float("nan"), "ego_y": float("nan"),
           "ego_angle": float("nan"), "ego_speed": float("nan"),
           "ego_lane": "", "neighbors": [], "tls": []}
    try:
        out["ego_x"], out["ego_y"] = libsumo_traci.vehicle.getPosition(EGO_ID)
        out["ego_angle"] = libsumo_traci.vehicle.getAngle(EGO_ID)
        out["ego_speed"] = libsumo_traci.vehicle.getSpeed(EGO_ID)
        out["ego_lane"] = libsumo_traci.vehicle.getLaneID(EGO_ID)
    except Exception:
        return out
    try:
        sub = libsumo_traci.vehicle.getContextSubscriptionResults(EGO_ID) or {}
        for nb_id, attrs in sub.items():
            if nb_id == EGO_ID:
                continue
            pos = attrs.get(libsumo_constants.VAR_POSITION)
            if pos is None:
                continue
            out["neighbors"].append((
                str(nb_id),
                float(pos[0]), float(pos[1]),
                float(attrs.get(libsumo_constants.VAR_ANGLE, 0.0)),
                float(attrs.get(libsumo_constants.VAR_SPEED, 0.0)),
                float(attrs.get(libsumo_constants.VAR_LENGTH, 5.0)),
            ))
    except Exception:
        pass
    return out


def run_episode(agent, env, base_env, seed, max_steps, libsumo_traci,
                libsumo_constants, EGO_ID, capture_traj=True):
    import numpy as np
    import torch
    import time

    base_env._rng = random.Random(seed)
    base_env._np_rng = np.random.default_rng(seed)
    obs = env.reset()
    obs_batch = {k: np.asarray(v)[None] for k, v in obs.items()}
    done_flag = np.zeros(1, dtype=bool)
    state = None
    ep_reward = 0.0
    ep_cost = 0.0
    term_reason = "timeout"
    steps = 0
    traj = {
        "actions": [], "rewards": [], "costs": [],
        "ego_x": [], "ego_y": [], "ego_angle": [], "ego_speed": [], "ego_lane": [],
        "nb_x": [], "nb_y": [], "nb_angle": [], "nb_speed": [], "nb_length": [],
    } if capture_traj else None

    t0 = time.time()
    with torch.no_grad():
        for t in range(max_steps):
            if traj is not None:
                snap = capture_state(libsumo_traci, libsumo_constants, EGO_ID)
                traj["ego_x"].append(snap["ego_x"])
                traj["ego_y"].append(snap["ego_y"])
                traj["ego_angle"].append(snap["ego_angle"])
                traj["ego_speed"].append(snap["ego_speed"])
                traj["ego_lane"].append(snap["ego_lane"])
                nb = snap["neighbors"]
                traj["nb_x"].append([n[1] for n in nb])
                traj["nb_y"].append([n[2] for n in nb])
                traj["nb_angle"].append([n[3] for n in nb])
                traj["nb_speed"].append([n[4] for n in nb])
                traj["nb_length"].append([n[5] for n in nb])
            policy_output, state = agent(obs_batch, done_flag, state=state, training=False)
            a_t = policy_output["action"]
            a_np = (a_t.detach().cpu().numpy()[0] if hasattr(a_t, "detach")
                    else np.asarray(a_t)[0])
            obs, r, done, info = env.step({"action": a_np})
            ep_reward += float(r)
            ep_cost += float(obs["cost"])
            steps = t + 1
            if traj is not None:
                traj["actions"].append(a_np.tolist())
                traj["rewards"].append(float(r))
                traj["costs"].append(float(obs["cost"]))
            if done:
                term_reason = info.get("term_reason", "done")
                break
            obs_batch = {k: np.asarray(v)[None] for k, v in obs.items()}
            done_flag = np.zeros(1, dtype=bool)
    return dict(
        seed=seed, steps=steps, reward=ep_reward, cost=ep_cost,
        sim_seconds=steps * 0.1, wall_seconds=time.time() - t0,
        term_reason=term_reason, traj=traj,
    )


def save_episode_npz(out_path: Path, ep, outcome):
    import numpy as np
    if ep["traj"] is None:
        return
    t = ep["traj"]
    np.savez_compressed(
        out_path,
        seed=ep["seed"], steps=ep["steps"], reward=ep["reward"], cost=ep["cost"],
        sim_seconds=ep["sim_seconds"], wall_seconds=ep["wall_seconds"],
        term_reason=ep["term_reason"], outcome=outcome,
        actions=np.array(t["actions"], dtype=np.float32),
        rewards=np.array(t["rewards"], dtype=np.float32),
        costs=np.array(t["costs"], dtype=np.float32),
        ego_x=np.array(t["ego_x"], dtype=np.float64),
        ego_y=np.array(t["ego_y"], dtype=np.float64),
        ego_angle=np.array(t["ego_angle"], dtype=np.float32),
        ego_speed=np.array(t["ego_speed"], dtype=np.float32),
        ego_lane=np.array(t["ego_lane"], dtype=object),
        nb_x=np.array(t["nb_x"], dtype=object),
        nb_y=np.array(t["nb_y"], dtype=object),
        nb_angle=np.array(t["nb_angle"], dtype=object),
        nb_speed=np.array(t["nb_speed"], dtype=object),
        nb_length=np.array(t["nb_length"], dtype=object),
    )


def aggregate(results):
    import numpy as np
    n = len(results)
    if n == 0:
        return {}
    outcomes = Counter(classify_outcome(r["term_reason"]) for r in results)
    rewards = np.array([r["reward"] for r in results])
    costs = np.array([r["cost"] for r in results])
    steps = np.array([r["steps"] for r in results])
    sim_s = np.array([r["sim_seconds"] for r in results])
    wall_s = np.array([r["wall_seconds"] for r in results])

    def rate(key): return outcomes.get(key, 0) / n

    return dict(
        n_episodes=n,
        success_rate=rate("arrival"),
        crash_fault_rate=rate("collision_fault"),
        crash_victim_rate=rate("collision_victim"),
        wrong_lane_rate=rate("wrong_lane_end"),
        stuck_rate=rate("stuck"),
        silent_stop_rate=rate("silent_stop"),
        drained_rate=rate("drained"),
        timeout_rate=rate("timeout"),
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        median_reward=float(np.median(rewards)),
        mean_cost=float(costs.mean()),
        std_cost=float(costs.std()),
        mean_steps=float(steps.mean()),
        std_steps=float(steps.std()),
        mean_sim_seconds=float(sim_s.mean()),
        mean_wall_seconds=float(wall_s.mean()),
        total_wall_seconds=float(wall_s.sum()),
        outcomes=dict(outcomes),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=15,
                   help="Number of eval episodes (each = 1 seed). Default 15.")
    p.add_argument("--seed-start", type=int, default=0,
                   help="Starting seed (seeds run = seed-start … seed-start+episodes-1)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Torch device. Default cpu (works everywhere).")
    p.add_argument("--max-steps", type=int, default=18000,
                   help="Hard step cap per episode (18000 = 30 min sim).")
    p.add_argument("--out-dir", default="outputs",
                   help="Where to write trajectories, summary, and video.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip the cinematic top-3 video render.")
    p.add_argument("--checkpoint", default=None,
                   help="Override path to checkpoint .pt. Default: auto-download.")
    p.add_argument("--heavy-traffic-window", default=f"{HEAVY_TRAFFIC_TMIN}-{HEAVY_TRAFFIC_TMAX}",
                   help="t_insert window (sim seconds) for heavy traffic, "
                        "as 'min-max'. Default 1500-2400 (peak Dallas).")
    args = p.parse_args()

    args.seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmin_s, tmax_s = (float(x) for x in args.heavy_traffic_window.split("-"))

    # Checkpoint
    ckpt_path = (Path(args.checkpoint).resolve() if args.checkpoint
                 else ensure_checkpoint(HERE))

    print()
    print("=" * 72)
    print("SUMO + DreamerV3/SafeDreamer Dallas city eval")
    print("=" * 72)
    print(f"  episodes:        {args.episodes}")
    print(f"  seeds:           {args.seeds[0]}–{args.seeds[-1]}")
    print(f"  device:          {args.device}")
    print(f"  checkpoint:      {ckpt_path}")
    print(f"  out_dir:         {out_dir}")
    print(f"  heavy traffic:   t_insert in [{tmin_s:.0f}s, {tmax_s:.0f}s] (peak rush)")
    print()

    # Build agent + env
    agent, env, base_env, config = build_agent_and_env(args, ckpt_path, out_dir)
    patch_heavy_traffic(base_env, tmin_s, tmax_s)

    # libsumo handles for trajectory capture
    try:
        import libsumo as _traci
        from libsumo import constants as _tc
    except ImportError:
        import traci as _traci
        from traci import constants as _tc
    from envs.sumo import EGO_ID

    results = []
    import time
    t_total = time.time()
    for i, seed in enumerate(args.seeds):
        print(f"[{i+1}/{len(args.seeds)}] seed={seed} … ", end="", flush=True)
        try:
            ep = run_episode(agent, env, base_env, seed, args.max_steps,
                             _traci, _tc, EGO_ID, capture_traj=True)
        except KeyboardInterrupt:
            print(" ABORTED by user")
            break
        outcome = classify_outcome(ep["term_reason"])
        print(f"steps={ep['steps']:>5}  ret={ep['reward']:+8.2f}  "
              f"cost={ep['cost']:6.2f}  sim={ep['sim_seconds']:6.1f}s  "
              f"wall={ep['wall_seconds']:5.1f}s  outcome={outcome}")
        save_episode_npz(out_dir / f"eval_seed{seed:03d}.npz", ep, outcome)
        ep_meta = {k: v for k, v in ep.items() if k != "traj"}
        ep_meta["outcome"] = outcome
        results.append(ep_meta)

    total_wall = time.time() - t_total
    print()
    print("=" * 72)
    summary = aggregate(results)
    summary["total_wall_seconds_full"] = total_wall
    summary["per_episode"] = results
    summary["checkpoint"] = str(ckpt_path)
    summary["device"] = args.device
    summary["heavy_traffic_window"] = [tmin_s, tmax_s]

    print(f"  Episodes:           {summary['n_episodes']}")
    print(f"  Success (arrival):  {summary['success_rate']*100:.1f}%  "
          f"({summary['outcomes'].get('arrival', 0)}/{summary['n_episodes']})")
    print(f"  Crash-at-fault:     {summary['crash_fault_rate']*100:.1f}%")
    print(f"  Crash-victim:       {summary['crash_victim_rate']*100:.1f}%")
    print(f"  Wrong-lane-end:     {summary['wrong_lane_rate']*100:.1f}%")
    print(f"  Drained / timeout:  "
          f"{(summary['drained_rate']+summary['timeout_rate'])*100:.1f}%")
    print()
    print(f"  Reward:    mean={summary['mean_reward']:+.2f}  "
          f"std={summary['std_reward']:.2f}  median={summary['median_reward']:+.2f}")
    print(f"  Cost:      mean={summary['mean_cost']:.2f}")
    print(f"  Steps:     mean={summary['mean_steps']:.0f}  std={summary['std_steps']:.0f}")
    print(f"  Wall time: total={total_wall:.0f}s")

    summary_path = out_dir / "eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary → {summary_path}")
    saved = sorted(out_dir.glob("eval_seed*.npz"))
    print(f"Saved {len(saved)} trajectory NPZ files in {out_dir}/")

    # Video render
    if not args.no_video:
        print()
        print("Building cinematic top-3 video (matplotlib + ffmpeg)…")
        video_script = HERE / "render_video.py"
        if not video_script.exists():
            print(f"  WARNING: {video_script.name} missing; skipping video.")
        else:
            try:
                subprocess.run(
                    [sys.executable, str(video_script),
                     "--eval-dir", str(out_dir),
                     "--out", str(out_dir / "top3_cinematic.mp4")],
                    check=True,
                )
                print(f"  → {out_dir / 'top3_cinematic.mp4'}")
            except subprocess.CalledProcessError as e:
                print(f"  video render failed (exit {e.returncode}); "
                      f"trajectories + summary are still saved.")

    try:
        env.close() if hasattr(env, "close") else None
    except Exception:
        pass
    print()
    print(f"DONE. All artifacts under: {out_dir}/")


if __name__ == "__main__":
    main()
