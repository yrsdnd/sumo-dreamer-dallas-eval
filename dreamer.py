import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
        # SafeDreamer PID-Lagrangian controller. When use_cost=False the
        # controller is inert (returns lagrange_penalty=0 ⇒ no ψ term ⇒
        # identical behaviour to upstream DreamerV3).
        self._use_cost = bool(getattr(config, "use_cost", False))
        self._pid = tools.PIDLagrangian(config) if self._use_cost else None
        # Round-5 (2026-04-30): adaptive cost-limit calibration. The world
        # model frequently under-predicts real-environment cost (mean imag
        # cost ≈ 0.05/step but real-cost rate often > 0.5/step on hot eps).
        # Tracking the gap as EMAs lets us tighten the imagined limit so
        # ψ stays informative. Both EMAs use 0.95 decay (≈20-event window).
        self._real_cost_rate_ema = 0.0     # per-step cost in real episodes
        self._imag_cost_rate_ema = 0.0     # per-imagined-step cost
        self._cost_rate_ema_alpha = 0.95
        self._cost_rate_ema_inited = False  # first sample replaces 0-init
        # 2026-05-02 (round-7): one-shot init telemetry so the operator can
        # confirm at startup that the agent + WM came up with the expected
        # config. Single line per construction event.
        try:
            n_params_wm = sum(p.numel() for p in self._wm.parameters())
            n_params_actor = sum(p.numel() for p in self._task_behavior.parameters())
            tools.DebugPrinter.header(
                "INIT-AGENT",
                device=str(getattr(config, "device", "?")),
                use_cost=self._use_cost,
                use_amp=getattr(config, "use_amp", False),
                wm_params=n_params_wm,
                actor_critic_params=n_params_actor,
                batch_size=getattr(config, "batch_size", "?"),
                batch_length=getattr(config, "batch_length", "?"),
                train_ratio=getattr(config, "train_ratio", "?"),
                imag_horizon=getattr(config, "imag_horizon", "?"),
                pid_penalty_max=(config.pid["penalty_max"] if self._use_cost else None),
                pid_init_pid_i=(self._pid._pid_i if self._use_cost else None),
            )
        except Exception as e:
            print(f"[DBG INIT-AGENT] (telemetry failed: {e!r})")

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    # Bug 1 fix (Dallas long-run, 2026-04-28):
                    # PID metrics (`pid_lagrange_penalty`, `pid_i`, ...)
                    # are populated only when an episode ends; between
                    # episodes the list is empty and `np.mean([])`
                    # returns NaN with a RuntimeWarning, polluting the
                    # log with cosmetic NaN entries. Now skip the log
                    # call when the list-typed metric is empty;
                    # scalar-typed metrics (e.g. `update_count`) still
                    # log normally.
                    if isinstance(values, list):
                        if not values:
                            continue
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                    else:
                        self._logger.scalar(name, float(values))
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if tools.DebugPrinter.every("policy", 200):
            tools.DebugPrinter.header(
                "POLICY", step=self._step, training=training, use_cost=self._use_cost
            )
            tools.DebugPrinter.stats("embed", embed)
            tools.DebugPrinter.stats("latent.stoch", latent["stoch"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        if tools.DebugPrinter.count("policy") % 200 == 0 and tools.DebugPrinter.enabled:
            tools.DebugPrinter.stats("action", action)
            tools.DebugPrinter.kv(logprob=logprob.mean())
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        # Read the current PID-Lagrangian multiplier; 0.0 when safety is off so
        # the actor loss is untouched relative to baseline DreamerV3.
        lagrange_penalty = self._pid.lagrange_penalty if self._pid is not None else 0.0
        # Round-5: compute adaptive effective_cost_limit. Sub-nominal when
        # real cost rate exceeds imagined cost rate (WM under-predicts
        # cost) so ψ goes positive even on benign-looking imag rollouts.
        effective_cost_limit = self._effective_cost_limit()
        beh_metrics = self._task_behavior._train(
            start,
            reward,
            lagrange_penalty=lagrange_penalty,
            effective_cost_limit=effective_cost_limit,
        )[-1]
        metrics.update(beh_metrics)
        # Update imag-side EMA from the per-batch rate that the behavior
        # train just surfaced. Done after the train so we use a real
        # measurement, not a stale init.
        imag_rate = beh_metrics.get("imag_cost_per_step", None)
        if imag_rate is not None and self._use_cost:
            self._update_imag_cost_rate_ema(float(imag_rate))
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _update_imag_cost_rate_ema(self, rate: float) -> None:
        """EMA-update the imagined per-step cost rate. Used by adaptive
        effective_cost_limit. First sample replaces the 0-init.
        """
        if not self._cost_rate_ema_inited:
            self._imag_cost_rate_ema = float(rate)
            # Initialize real EMA as well so the gap_ratio starts at 1.0
            # and the effective limit isn't tightened on a single sample.
            if self._real_cost_rate_ema == 0.0:
                self._real_cost_rate_ema = float(rate)
            self._cost_rate_ema_inited = True
        else:
            a = self._cost_rate_ema_alpha
            self._imag_cost_rate_ema = a * self._imag_cost_rate_ema + (1.0 - a) * float(rate)

    def _update_real_cost_rate_ema(self, episode_cost: float, episode_length: int) -> None:
        """EMA-update the real per-step cost rate. Called once per episode.
        """
        if episode_length <= 0:
            return
        rate = float(episode_cost) / float(episode_length)
        if not self._cost_rate_ema_inited:
            # Real-EMA arrives before imag-EMA on the very first episode of
            # a fresh run (prefill). Stash; let imag-EMA init copy us.
            self._real_cost_rate_ema = rate
            return
        a = self._cost_rate_ema_alpha
        self._real_cost_rate_ema = a * self._real_cost_rate_ema + (1.0 - a) * rate

    def _effective_cost_limit(self) -> float:
        """Layer 2 adaptive cost-limit. When real cost rate exceeds imagined
        cost rate (WM under-predicts cost), tighten the imagined-side limit
        so ψ goes positive on benign-looking imagined rollouts.

        Safety: gap_ratio is clipped to [1.0, 10.0] — never relaxes the
        limit, never tightens beyond 10×. Returns nominal cost_limit until
        both EMAs are initialised.
        """
        if not self._use_cost or not self._cost_rate_ema_inited:
            return float(self._config.cost_limit)
        if self._imag_cost_rate_ema <= 1e-3:
            return float(self._config.cost_limit)
        gap_ratio = self._real_cost_rate_ema / max(self._imag_cost_rate_ema, 1e-3)
        gap_ratio = float(np.clip(gap_ratio, 1.0, 10.0))
        return float(self._config.cost_limit) / gap_ratio

    def on_episode_end(self, episode_cost: float, episode_length: int = 0):
        """Call after each real-environment episode so the PID-Lagrangian
        controller can update its multiplier using the total episode cost.
        tools.simulate invokes this through the callback it receives.

        Round-5 (2026-04-30): also takes episode_length so the real cost
        rate (per-step) can be EMA-tracked for the adaptive cost-limit.
        """
        # Real-side EMA update happens regardless of PID presence — we may
        # eventually use it for diagnostics in baseline runs too.
        self._update_real_cost_rate_ema(episode_cost, episode_length)
        if self._pid is None:
            # Baseline (use_cost=False): still log the episode cost so we can
            # compare against the safe run.
            tools.DebugPrinter.header("EPISODE", episode_cost=episode_cost, pid=False)
            return
        pen, pid_d, pid_i, delta_p = self._pid.pid_update(episode_cost, self._step)
        # Surface to tensorboard via the standard metrics dict
        self._metrics.setdefault("pid_lagrange_penalty", []).append(pen)
        self._metrics.setdefault("pid_i", []).append(pid_i)
        self._metrics.setdefault("pid_d", []).append(pid_d)
        self._metrics.setdefault("pid_delta_p", []).append(delta_p)
        self._metrics.setdefault("episode_cost", []).append(float(episode_cost))
        # Round-5 diagnostics — surface both EMA tracks + the effective
        # limit so the metric log reveals when the gap is opening up.
        self._metrics.setdefault("real_cost_rate_ema", []).append(float(self._real_cost_rate_ema))
        self._metrics.setdefault("imag_cost_rate_ema", []).append(float(self._imag_cost_rate_ema))
        self._metrics.setdefault("effective_cost_limit", []).append(float(self._effective_cost_limit()))
        tools.DebugPrinter.header("EPISODE", episode_cost=episode_cost, pid=True)
        tools.DebugPrinter.kv(
            lagrange_penalty=pen,
            pid_i=pid_i,
            pid_d=pid_d,
            delta_p=delta_p,
            cost_limit=self._pid.current_cost_limit,
            effective_cost_limit=self._effective_cost_limit(),
            real_rate=self._real_cost_rate_ema,
            imag_rate=self._imag_cost_rate_ema,
        )


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "sumo":
        import envs.sumo as sumo_env

        env = sumo_env.SumoEnv(task, seed=config.seed + id)
        # Action is already in [-1, 1]; no NormalizeActions/OneHotAction needed.
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    # Round-6 perf (2026-05-02): only spawn parallel eval-env subprocesses
    # when eval_episode_num > 0. With eval_episode_num=0 (current sumo
    # config) the eval envs are NEVER used for training — they were
    # consuming ~1 GB of RAM and 4 idle background subprocesses for
    # nothing. The eval-env objects are still created above so the
    # variable is defined (the eval block at line 521 is also gated by
    # eval_episode_num > 0, so eval_envs being unwrapped is fine).
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        if config.eval_episode_num > 0:
            eval_envs = [Parallel(env, "process") for env in eval_envs]
        else:
            eval_envs = [Damy(env) for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        # Round-6 (2026-05-02): map_location=config.device so a CPU-saved
        # checkpoint (step 21k) loads directly onto MPS without an
        # intermediate CPU→MPS hop during load_state_dict's tensor copy.
        # Identity behaviour when device='cpu'.
        checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        # Restore PID controller state so λ_p doesn't restart from 0
        if agent._pid is not None and "pid_state_dict" in checkpoint:
            try:
                agent._pid.load_state_dict(checkpoint["pid_state_dict"])
                print(
                    f"Restored PID: λ_p={agent._pid.lagrange_penalty:.4f} "
                    f"I={agent._pid.pid_i:.4f}"
                )
            except Exception as e:
                print(f"Warning: could not restore PID state: {e!r}")

    # Wallclock-time termination (added 2026-04-28 for the Dallas long-run).
    # When `config.max_train_hours > 0`, the main loop exits after that many
    # real hours irrespective of step count. Useful for "train for 1.5 days"
    # workflows where the user can't predict step throughput.
    import time as _time
    import threading as _threading
    _start_time = _time.time()
    _max_secs = (
        float(getattr(config, "max_train_hours", 0.0)) * 3600.0
        if getattr(config, "max_train_hours", 0.0) > 0 else float("inf")
    )
    if _max_secs < float("inf"):
        print(f"[wall-clock] training will exit after {_max_secs/3600:.1f} h "
              f"({_max_secs:.0f} s)")

    # Bug 2 fix (Dallas long-run, 2026-04-28): the progress print used to be
    # inside the outer `while agent._step < ...` loop body. That body only
    # iterates AFTER `tools.simulate(...)` returns from each `eval_every`
    # chunk (~30 min), so progress lines fired at most every 30 min and
    # often missed because the 60-s threshold check happened only between
    # chunks. Switching to a daemon thread that prints every 60 s
    # regardless of where main() is in the call stack — survives blocking
    # SUMO calls, blocking gradient steps, etc.
    _shutdown_progress = _threading.Event()
    def _progress_loop():
        while not _shutdown_progress.is_set():
            if _shutdown_progress.wait(timeout=60.0):
                return
            try:
                elapsed = _time.time() - _start_time
                step_now = int(agent._step)
                if _max_secs < float("inf"):
                    pct = min(99.99, 100.0 * elapsed / _max_secs)
                    eta_h = max(0.0, (_max_secs - elapsed) / 3600.0)
                    print(
                        f"[progress] {pct:5.2f}% | elapsed {elapsed/3600:.2f} h"
                        f" | remaining {eta_h:.2f} h | step {step_now:>7d} /"
                        f" {int(config.steps)}",
                        flush=True,
                    )
                else:
                    pct = 100.0 * step_now / max(1, int(config.steps))
                    print(
                        f"[progress] step {step_now:>7d} /"
                        f" {int(config.steps)}  ({pct:5.2f}%)"
                        f" | elapsed {elapsed/3600:.2f} h",
                        flush=True,
                    )
            except Exception:
                pass  # never crash the trainer because of a print
    _progress_thread = _threading.Thread(target=_progress_loop, daemon=True)
    _progress_thread.start()

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        # Exit if wallclock budget exceeded.
        if (_time.time() - _start_time) >= _max_secs:
            print(f"[wall-clock] {_max_secs/3600:.1f} h budget reached at "
                  f"step {agent._step}; exiting main loop")
            _shutdown_progress.set()
            break

        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "config": vars(config),
            "step": agent._step,
        }
        # Persist PID controller so λ_p / integrator / derivative EMA survive
        # restart. Without this the safety-learning arm resets to scratch.
        if agent._pid is not None:
            items_to_save["pid_state_dict"] = agent._pid.state_dict()
        torch.save(items_to_save, logdir / "latest.pt")
        # Keep only the last KEEP milestone checkpoints to bound disk use.
        # Each checkpoint is ~300 MB for a 26M-param agent; KEEP=3 caps
        # the accumulated milestone usage at ~900 MB.
        KEEP = int(getattr(config, "keep_checkpoints", 3))
        try:
            if KEEP > 0:
                torch.save(
                    items_to_save, logdir / f"checkpoint_step_{agent._step:07d}.pt"
                )
                # Delete older checkpoints beyond the retention window.
                mk = sorted(
                    logdir.glob("checkpoint_step_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                for stale in mk[:-KEEP]:
                    try:
                        stale.unlink()
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: milestone checkpoint save failed: {e!r}")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    # Newer ruamel.yaml removed the module-level safe_load helper; use the
    # YAML object directly so this works across 0.17+ and 0.18+.
    _yaml = yaml.YAML(typ="safe", pure=True)
    configs = _yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
