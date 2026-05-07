import datetime
import collections
import io
import os
import json
import pathlib
import re
import threading
import time
import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter


to_np = lambda x: x.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Debug instrumentation: throttled prints for sanity-checking a live run.
# Toggle with env var SUMO_DBG=0 to silence, SUMO_DBG=1 (default) to enable.
# Interval keys let each call site throttle independently.
# ---------------------------------------------------------------------------
class DebugPrinter:
    """Centralised throttled debug output with optional tensor stats.

    Usage
    -----
    >>> if DebugPrinter.every("wm_train", 20):
    ...     DebugPrinter.header("WM", step=step)
    ...     DebugPrinter.stats("post.stoch", post["stoch"])
    """

    _counters: dict = {}
    enabled: bool = os.environ.get("SUMO_DBG", "1") not in ("0", "false", "False")

    @classmethod
    def every(cls, key: str, n: int) -> bool:
        c = cls._counters.get(key, 0)
        cls._counters[key] = c + 1
        return cls.enabled and (c % n == 0)

    @classmethod
    def count(cls, key: str) -> int:
        return cls._counters.get(key, 0)

    @classmethod
    def header(cls, tag: str, **kv):
        if not cls.enabled:
            return
        bits = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"[DBG {tag}] {bits}", flush=True)

    @classmethod
    def stats(cls, label: str, x, full: bool = False):
        """Print shape / dtype / min / max / mean / std of x (numpy or torch)."""
        if not cls.enabled:
            return
        try:
            if hasattr(x, "detach"):
                x_np = x.detach().cpu().numpy()
            elif isinstance(x, (list, tuple)):
                import torch as _t
                if len(x) > 0 and hasattr(x[0], "detach"):
                    x_np = _t.stack(x).detach().cpu().numpy()
                else:
                    x_np = np.asarray(x)
            else:
                x_np = np.asarray(x)
            x_np = x_np.astype(np.float32, copy=False) if x_np.dtype != np.float32 else x_np
            finite = np.isfinite(x_np)
            if finite.all():
                mn, mx, me, sd = (
                    float(x_np.min()),
                    float(x_np.max()),
                    float(x_np.mean()),
                    float(x_np.std()),
                )
            else:
                mn, mx, me, sd = (float("nan"),) * 4
            print(
                f"    [{label}] shape={tuple(x_np.shape)} dtype={x_np.dtype} "
                f"min={mn:+.4f} max={mx:+.4f} mean={me:+.4f} std={sd:.4f}"
                + (" NANS" if not finite.all() else ""),
                flush=True,
            )
            if full and x_np.size <= 12:
                print(f"      values={x_np.flatten().tolist()}", flush=True)
        except Exception as e:
            print(f"    [{label}] (stats-error: {e!r})", flush=True)

    @classmethod
    def kv(cls, **kv):
        if not cls.enabled:
            return
        bits = " ".join(f"{k}={cls._fmt(v)}" for k, v in kv.items())
        print(f"    {bits}", flush=True)

    @classmethod
    def _fmt(cls, v):
        if isinstance(v, float):
            return f"{v:+.4f}"
        if hasattr(v, "detach"):
            try:
                return f"{float(v.detach().cpu().item()):+.4f}"
            except Exception:
                pass
        return str(v)


class PIDLagrangian:
    """PID-controlled Lagrangian multiplier for constrained RL (Stooke et al., 2020).

    Ported from SafeDreamer (PKU-Alignment) `SafeDreamer/train.py:PIDLagrangian`.
    Pure numpy; no gradients. Call ``pid_update(episode_cost, step)`` once per
    finished real-environment episode; read ``lagrange_penalty`` whenever the
    actor loss wants the current multiplier.

    Config (dict accessed by key, matches DreamerV3-torch argparse.Namespace style):
        config.cost_limit: float, target cost threshold d.
        config.pid: dict with keys
            kp, ki, kd: PID gains.
            d_delay: int, lag of the derivative term (steps).
            delta_p_ema_alpha, delta_d_ema_alpha: EMA smoothing for P and D.
            penalty_max: cap on the multiplier.
            sum_norm, diff_norm: optional normalisation flags.
            lagrangian_multiplier_init: initial integral term.
            init_penalty: initial multiplier value.
            use_cost_decay: bool.
            init_cost_limit, decay_time_step, decay_num, decay_limit_step:
                cost-limit schedule (only when use_cost_decay=True).
    """

    def __init__(self, config):
        pid = config.pid
        self._kp = pid["kp"]
        self._ki = pid["ki"]
        self._kd = pid["kd"]
        self._d_delay = pid["d_delay"]
        self._a_p = pid["delta_p_ema_alpha"]
        self._a_d = pid["delta_d_ema_alpha"]
        self._penalty_max = pid["penalty_max"]
        self._sum_norm = bool(pid.get("sum_norm", True))
        self._diff_norm = bool(pid.get("diff_norm", False))
        self._cost_limit = float(config.cost_limit)
        self._pid_i = float(pid.get("lagrangian_multiplier_init", 0.0))
        self._cost_penalty = float(pid.get("init_penalty", 0.0))
        self._delta_p = 0.0
        self._cost_d = 0.0
        self._pid_d = 0.0
        self._cost_ds = collections.deque([0.0], maxlen=max(1, self._d_delay))
        self._use_cost_decay = bool(pid.get("use_cost_decay", False))
        self._current_cost_limit = float(pid.get("init_cost_limit", self._cost_limit))
        if self._use_cost_decay:
            dts = pid["decay_time_step"]
            dn = pid["decay_num"]
            dls = pid["decay_limit_step"]
            icl = pid["init_cost_limit"]
            self._decay_steps = [dts * (i + 1) for i in range(dn)]
            self._decay_limits = [max(icl - i * dls, self._cost_limit) for i in range(dn)]
        # Round-6 (2026-05-02): saturation diagnostics. The clip at the end of
        # pid_update() silently truncates pid_o > penalty_max. Previously this
        # made it look like λ "naturally" hit 0.5 — actually the controller
        # wanted more push than the cap allows. Track this explicitly so the
        # log shows when the cap is the bottleneck rather than convergence.
        self._call_count = 0
        self._saturation_streak = 0
        self._max_pid_o_seen = 0.0

    @property
    def lagrange_penalty(self) -> float:
        return float(self._cost_penalty)

    @property
    def delta_p(self) -> float:
        return float(self._delta_p)

    @property
    def pid_i(self) -> float:
        return float(self._pid_i)

    @property
    def pid_d(self) -> float:
        return float(self._pid_d)

    @property
    def current_cost_limit(self) -> float:
        return float(self._current_cost_limit)

    def pid_update(self, episode_cost: float, step: int):
        """Update controller with an episode's total cost; returns
        (cost_penalty, pid_d, pid_i, delta_p).
        """
        self._call_count += 1
        ep = float(episode_cost)
        if self._use_cost_decay:
            self._current_cost_limit = self._cost_limit
            for thr, lim in zip(self._decay_steps, self._decay_limits):
                if step < thr:
                    self._current_cost_limit = lim
                    break
        else:
            self._current_cost_limit = self._cost_limit

        delta = ep - self._current_cost_limit
        # Integral term (non-negative; optional [0,1] clamp under diff_norm)
        self._pid_i = max(0.0, self._pid_i + delta * self._ki)
        if self._diff_norm:
            self._pid_i = float(np.clip(self._pid_i, 0.0, 1.0))
        # Proportional EMA
        self._delta_p = self._a_p * self._delta_p + (1.0 - self._a_p) * delta
        # Derivative EMA on raw episode cost, with d_delay-step look-back
        self._cost_d = self._a_d * self._cost_d + (1.0 - self._a_d) * ep
        self._pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._kp * self._delta_p + self._pid_i + self._kd * self._pid_d
        # Round-6 (2026-05-02): record uncapped pid_o BEFORE clipping so we can
        # detect saturation (controller wants > penalty_max but cap silently
        # clips). delta>0 means cost is above target so the controller should
        # be pushing UP; if pid_o>=cap and delta>0, the cap is the bottleneck.
        pid_o_uncapped = float(pid_o)
        self._max_pid_o_seen = max(self._max_pid_o_seen, pid_o_uncapped)
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
        self._cost_penalty = float(np.clip(self._cost_penalty, 0.0, self._penalty_max))
        # Saturation = controller hit cap AND still wants to push (delta > 0,
        # i.e. recent ep_cost still above target_limit).
        saturated = (pid_o_uncapped >= self._penalty_max) and (delta > 0.0)
        if saturated:
            self._saturation_streak += 1
        else:
            self._saturation_streak = 0
        # Verbose per-update debug. Always prints — one line per finished real
        # episode, low volume. ⚠ flag marks consecutive saturation so the user
        # can spot "λ stuck at cap" early instead of after metrics tell the tale.
        warn = (
            "  ⚠ STUCK AT CAP — raise pid.penalty_max"
            if self._saturation_streak >= 5 else (
                f"  [SATURATED streak={self._saturation_streak}]"
                if saturated else ""
            )
        )
        print(
            f"[PID #{self._call_count:04d} @step {step:>7d}] "
            f"ep_cost={ep:7.2f}  limit={self._current_cost_limit:5.2f}  "
            f"delta={delta:+7.3f}  |  "
            f"P={self._kp * self._delta_p:+6.3f}  "
            f"I={self._pid_i:6.3f}  "
            f"D={self._kd * self._pid_d:+6.3f}  |  "
            f"pid_o={pid_o_uncapped:6.3f}  "
            f"λ={self._cost_penalty:.3f}/{self._penalty_max:.2f}"
            f"{warn}",
            flush=True,
        )
        return self._cost_penalty, self._pid_d, self._pid_i, self._delta_p

    def state_dict(self):
        return {
            "_pid_i": self._pid_i,
            "_cost_penalty": self._cost_penalty,
            "_delta_p": self._delta_p,
            "_cost_d": self._cost_d,
            "_pid_d": self._pid_d,
            "_cost_ds": list(self._cost_ds),
            "_current_cost_limit": self._current_cost_limit,
        }

    def load_state_dict(self, state):
        self._pid_i = float(state["_pid_i"])
        self._cost_penalty = float(state["_cost_penalty"])
        self._delta_p = float(state["_delta_p"])
        self._cost_d = float(state["_cost_d"])
        self._pid_d = float(state["_pid_d"])
        self._cost_ds = collections.deque(
            [float(x) for x in state["_cost_ds"]], maxlen=max(1, self._d_delay)
        )
        self._current_cost_limit = float(state["_current_cost_limit"])


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                # Episode cost (SafeDreamer): sum the per-step cost signal if
                # the env emitted one. Absent for non-safety envs → sum is 0.
                if "cost" in cache[envs[i].id]:
                    episode_cost = float(np.array(cache[envs[i].id]["cost"]).sum())
                else:
                    episode_cost = 0.0
                # 2026-05-02 (M2 fix): cap episode cost passed to the PID
                # controller. Without this cap a single pathological episode
                # (e.g. ego stuck on a red-light edge for 100 steps + a
                # collision = cost ~172) sends delta_p = ep - cost_limit = 162
                # into the controller, well outside its design range. The
                # cap (50, ≈ 5× the cost_limit of 10) preserves the rare-bad-
                # episode signal without overwhelming the integral term and
                # without changing the per-step cost the world-model trains on.
                EP_COST_CAP = 50.0
                if episode_cost > EP_COST_CAP:
                    episode_cost = EP_COST_CAP
                # Feed the completed-episode cost into the agent's PID-Lagrangian
                # controller, if it exposes the optional callback.
                # Round-5 (2026-04-30): also pass episode length so the agent
                # can compute per-step real cost rate for adaptive cost-limit
                # calibration. Wrapped in a try around the new signature so
                # older agents with the 1-arg signature still work.
                if hasattr(agent, "on_episode_end"):
                    try:
                        try:
                            agent.on_episode_end(episode_cost, length)
                        except TypeError:
                            agent.on_episode_end(episode_cost)
                    except Exception as e:
                        print(f"Warning: agent.on_episode_end raised {e!r}")
                # Some envs (e.g. numeric-only SUMO) do not produce an image;
                # avoid KeyError here and skip video logging in that case.
                video = cache[envs[i].id].get("image", None)
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.scalar(f"train_episode_cost", episode_cost)
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    """Save episodes as NPZ files.

    Round-6 perf (2026-05-02): the disk write happens in a daemon
    BACKGROUND thread so the simulate() main loop doesn't block on I/O.
    The trainer reads from in-memory `cache` (in tools.simulate), not
    from disk, so async writes don't affect training data quality —
    only the moment the file lands on disk shifts by a few ms.

    Each episode filename is unique (timestamp + uuid + length), so
    parallel daemon threads cannot race on the same file. We use
    `daemon=True` so the main process can exit cleanly even if a
    write is in flight; in that case we lose at most the last
    in-flight episode file (the data already used for training is
    preserved in agent state).
    """
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    # Take a shallow snapshot reference so the background thread sees
    # the dict the caller passed; the caller is expected not to mutate
    # the episode arrays after handing them here (matches existing
    # contract — caller calls save_episodes then moves on).
    snap = dict(episodes)

    def _do_write():
        for filename, episode in snap.items():
            length = len(episode["reward"])
            filepath = directory / f"{filename}-{length}.npz"
            try:
                with io.BytesIO() as f1:
                    np.savez_compressed(f1, **episode)
                    f1.seek(0)
                    with filepath.open("wb") as f2:
                        f2.write(f1.read())
            except Exception as e:
                # Background thread errors should not crash main process
                print(f"Warning: async save_episodes failed for {filepath}: {e!r}")

    threading.Thread(target=_do_write, daemon=True).start()
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        # 2026-05-02 (round-7 follow-up): same migration as M10 autocast.
        # No-op when use_amp=False (default at precision=32) so this only
        # silences the FutureWarning; behavior is identical.
        # 2026-05-07 (audit fix): GradScaler is CUDA-only in current PyTorch.
        # On macOS/MPS the existing line was silently a no-op even with
        # use_amp=True; force enabled=False on non-CUDA so future precision=16
        # runs on MPS don't pretend to be scaling gradients.
        _scaler_enabled = bool(use_amp) and torch.cuda.is_available()
        self._scaler = torch.amp.GradScaler("cuda", enabled=_scaler_enabled)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
