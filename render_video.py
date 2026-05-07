"""Cinematic video renderer — reads per-episode trajectory NPZs from
`run.py`, picks the top-N longest *arrival* episodes, and writes ONE
combined MP4 with a camera intro per episode (start zoomed-out covering
the full Dallas city, smoothly zoom in to ego close-up over ~5 seconds,
then track ego). Pure matplotlib + ffmpeg — no display server required.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# sumolib path setup (works on Linux/macOS/Windows when SUMO is installed)
_SUMO_HOME = os.environ.get("SUMO_HOME")
if _SUMO_HOME:
    tools_dir = os.path.join(_SUMO_HOME, "tools")
    if os.path.isdir(tools_dir) and tools_dir not in sys.path:
        sys.path.append(tools_dir)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib import animation

try:
    import sumolib
except ImportError:
    print("ERROR: sumolib not importable. Set SUMO_HOME env var or install "
          "the eclipse-sumo Python package: pip install eclipse-sumo")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
DEFAULT_NET = HERE / "envs" / "sumo_city_dallas" / "osm.net.xml.gz"

# Cinematic camera params
INTRO_FRAMES = 50          # ≈5 seconds intro at sim 10 Hz
CLOSE_RADIUS_M = 80.0      # half-window around ego after intro
FPS = 20
FRAME_STRIDE = 2
DPI = 130
EGO_W, EGO_L = 1.9, 4.6
NB_W = 1.9


def load_lanes(net_path: Path):
    print(f"Loading SUMO net (5–15 s)…")
    net = sumolib.net.readNet(str(net_path), withInternal=False)
    lanes = []
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        for lane in edge.getLanes():
            lanes.append(lane.getShape())
    xs = [p[0] for shape in lanes for p in shape]
    ys = [p[1] for shape in lanes for p in shape]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    print(f"  loaded {len(lanes)} lanes; bbox "
          f"{bbox[2]-bbox[0]:.0f}×{bbox[3]-bbox[1]:.0f} m")
    return lanes, bbox


def angle_to_rad(deg_north_cw):
    return np.deg2rad(90.0 - deg_north_cw)


def draw_vehicle(ax, x, y, angle_deg, length, width, color,
                 ec="black", lw=0.5, zorder=3):
    rad = angle_to_rad(angle_deg)
    rect = Rectangle((-length, -width / 2), length, width, color=color,
                     ec=ec, lw=lw, zorder=zorder)
    tr = Affine2D().rotate(rad).translate(x, y)
    rect.set_transform(tr + ax.transData)
    ax.add_patch(rect)


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def render_episode(npz_path: Path, lanes, bbox, frame_dir: Path,
                   episode_index: int, total: int):
    print(f"  Rendering episode {episode_index}/{total}: {npz_path.name}")
    with np.load(npz_path, allow_pickle=True) as d:
        ego_x = d["ego_x"]; ego_y = d["ego_y"]
        ego_angle = d["ego_angle"]; ego_speed = d["ego_speed"]
        nb_x = d["nb_x"]; nb_y = d["nb_y"]
        nb_angle = d["nb_angle"]; nb_length = d["nb_length"]
        rewards = d["rewards"]; costs = d["costs"]; actions = d["actions"]
        seed = int(d["seed"]); outcome = str(d["outcome"])
        term_reason = str(d["term_reason"])
        steps = int(d["steps"])
        ep_reward = float(d["reward"]); ep_cost = float(d["cost"])

    T = min(len(ego_x), len(rewards))
    cx_city = (bbox[0] + bbox[2]) / 2.0
    cy_city = (bbox[1] + bbox[3]) / 2.0
    city_half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0 * 1.05

    valid = ~np.isnan(ego_x)
    path_x = ego_x[valid]; path_y = ego_y[valid]

    frames = sorted(set(list(range(0, INTRO_FRAMES, 1)) +
                        list(range(INTRO_FRAMES, T, FRAME_STRIDE))))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    n_written = 0

    for t in frames:
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        if t < INTRO_FRAMES and not np.isnan(ego_x[t]):
            f = smoothstep(t / max(INTRO_FRAMES - 1, 1))
            radius = city_half + f * (CLOSE_RADIUS_M - city_half)
            cx = cx_city + f * (float(ego_x[t]) - cx_city)
            cy = cy_city + f * (float(ego_y[t]) - cy_city)
        elif not np.isnan(ego_x[t]):
            cx, cy, radius = float(ego_x[t]), float(ego_y[t]), CLOSE_RADIUS_M
        else:
            cx, cy, radius = cx_city, cy_city, city_half

        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)

        line_w = max(0.4, min(2.5, 60.0 / radius))
        xmin, xmax = cx - radius, cx + radius
        ymin, ymax = cy - radius, cy + radius
        for shape in lanes:
            xs = [p[0] for p in shape]; ys = [p[1] for p in shape]
            if max(xs) < xmin or min(xs) > xmax or max(ys) < ymin or min(ys) > ymax:
                continue
            ax.plot(xs, ys, color="0.7", lw=line_w, alpha=0.65,
                    solid_capstyle="round", zorder=1)

        ax.plot(path_x, path_y, color="C0", lw=1.2, alpha=0.45, zorder=2)

        if radius < city_half * 0.6:
            for k in range(len(nb_x[t])):
                draw_vehicle(ax, nb_x[t][k], nb_y[t][k], nb_angle[t][k],
                             nb_length[t][k], NB_W, "0.4", zorder=3)

        if not np.isnan(ego_x[t]):
            if radius > 200:
                ax.scatter([float(ego_x[t])], [float(ego_y[t])], s=80,
                           color="gold", ec="black", lw=0.8, zorder=5)
            else:
                draw_vehicle(ax, float(ego_x[t]), float(ego_y[t]),
                             float(ego_angle[t]), EGO_L, EGO_W,
                             "gold", ec="black", lw=0.8, zorder=4)

        sim_t = t * 0.1
        title = (f"[{episode_index}/{total}]  seed={seed:03d}  "
                 f"step={t:5d}/{steps}  sim_t={sim_t:6.1f}s  outcome={outcome}")
        ax.set_title(title, fontsize=11)
        a = actions[min(t, len(actions) - 1)]
        info = (f"radius={radius:5.0f} m\n"
                f"ego_speed={float(ego_speed[t]):5.2f} m/s\n"
                f"action=[{a[0]:+.2f}, {a[1]:+.2f}]\n"
                f"step_reward={float(rewards[min(t, len(rewards)-1)]):+7.3f}\n"
                f"step_cost  ={float(costs[min(t, len(costs)-1)]):.2f}\n"
                f"ep_reward  ={ep_reward:+7.2f}\n"
                f"ep_cost    ={ep_cost:6.2f}\n"
                f"term       ={term_reason}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8.5,
                family="monospace", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="0.7"))
        fig.savefig(frame_dir / f"frame_{n_written:06d}.png",
                    dpi=DPI, bbox_inches="tight")
        n_written += 1

    plt.close(fig)
    return n_written


def encode_video(frame_dir: Path, out_mp4: Path, fps: int = FPS):
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", str(frame_dir / "frame_%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           str(out_mp4)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def concat_videos(parts, out_mp4: Path):
    list_path = out_mp4.parent / "_concat_list.txt"
    with list_path.open("w") as f:
        for v in parts:
            f.write(f"file '{v.resolve()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_path), "-c", "copy", str(out_mp4)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    list_path.unlink(missing_ok=True)


def pick_top_arrivals(eval_dir: Path, n: int):
    files = sorted(eval_dir.glob("eval_seed*.npz"))
    cands = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            cands.append({"path": f, "seed": int(d["seed"]),
                          "outcome": str(d["outcome"]),
                          "steps": int(d["steps"]),
                          "reward": float(d["reward"])})
    arrivals = sorted([c for c in cands if c["outcome"] == "arrival"],
                      key=lambda c: -c["steps"])
    return arrivals[:n], cands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--net", default=str(DEFAULT_NET))
    p.add_argument("--top-n", type=int, default=3)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top, all_eps = pick_top_arrivals(eval_dir, args.top_n)
    if not top:
        print(f"WARNING: no arrival episodes in {eval_dir}; "
              f"can't render the top-N video.")
        sys.exit(0)
    print(f"\nFound {len(all_eps)} episodes, picking top-{len(top)} arrivals:")
    for e in top:
        print(f"  seed={e['seed']:>3}  steps={e['steps']:>5}  "
              f"reward={e['reward']:+8.2f}")

    lanes, bbox = load_lanes(Path(args.net))

    work = Path("/tmp" if os.name == "posix" else os.environ.get("TEMP", ".")) / "sumo_cinematic"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    parts = []
    for i, ep in enumerate(top, 1):
        ep_dir = work / f"ep{i:02d}"
        ep_dir.mkdir(parents=True)
        n = render_episode(ep["path"], lanes, bbox, ep_dir, i, len(top))
        ep_video = work / f"ep{i:02d}.mp4"
        encode_video(ep_dir, ep_video)
        parts.append(ep_video)
        print(f"    encoded {n} frames")

    concat_videos(parts, out_path)
    print(f"\n✓ Combined video → {out_path}")


if __name__ == "__main__":
    main()
