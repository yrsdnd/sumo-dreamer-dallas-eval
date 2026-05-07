import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        # SafeDreamer cost head (optional, gated by config.use_cost). Matches the
        # safedreamer_v3_complete.html spec: 5 layers × 1024 units, symlog_disc
        # DiscDist with 255 bins.
        if getattr(config, "use_cost", False):
            self.heads["cost"] = networks.MLP(
                feat_size,
                (255,) if config.cost_head["dist"] == "symlog_disc" else (),
                config.cost_head["layers"],
                config.cost_head.get("units", config.units),
                config.act,
                config.norm,
                dist=config.cost_head["dist"],
                outscale=config.cost_head["outscale"],
                device=config.device,
                name="Cost",
            )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        if getattr(config, "use_cost", False):
            self._scales["cost"] = config.cost_head["loss_scale"]
        # 2026-05-02 (round-7): WorldModel construction telemetry. Lets
        # the operator confirm encoder/decoder/RSSM dims at startup.
        try:
            tools.DebugPrinter.header(
                "INIT-WM",
                obs_keys=sorted(shapes.keys()),
                obs_shapes={k: tuple(v) for k, v in shapes.items()},
                embed_size=self.embed_size,
                feat_size=feat_size,
                dyn_stoch=config.dyn_stoch,
                dyn_discrete=config.dyn_discrete,
                dyn_deter=config.dyn_deter,
                heads=sorted(self.heads.keys()),
                use_cost=bool(getattr(config, "use_cost", False)),
            )
        except Exception as e:
            print(f"[DBG INIT-WM] (telemetry failed: {e!r})")

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        # Debug: show replay batch shapes + reward/cost/terminal stats
        if tools.DebugPrinter.every("wm_batch", 100):  # was 20 (round-5)
            tools.DebugPrinter.header("WM-BATCH", call=tools.DebugPrinter.count("wm_batch"))
            for k in sorted(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    tools.DebugPrinter.stats(f"data.{k}", data[k])

        with tools.RequiresGrad(self):
            with torch.amp.autocast(
                device_type=str(self._config.device).split(":")[0],
                enabled=self._use_amp,
            ):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                # Compare real posterior vs predicted prior (dreaming accuracy signal)
                if tools.DebugPrinter.every("wm_rssm", 100):  # was 20 (round-5)
                    tools.DebugPrinter.header("WM-RSSM")
                    tools.DebugPrinter.stats("embed", embed)
                    tools.DebugPrinter.stats("post.stoch", post["stoch"])
                    tools.DebugPrinter.stats("prior.stoch", prior["stoch"])
                    tools.DebugPrinter.stats("post.deter", post["deter"])
                    tools.DebugPrinter.stats("kl_value", kl_value)
                    try:
                        post_flat = post["stoch"].reshape(*post["stoch"].shape[:2], -1)
                        prior_flat = prior["stoch"].reshape(*prior["stoch"].shape[:2], -1)
                        drift = torch.linalg.norm(post_flat - prior_flat, dim=-1).mean()
                        tools.DebugPrinter.kv(post_vs_prior_L2=drift)
                    except Exception:
                        pass
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    # SafeDreamer: upweight loss on positive-cost steps so the
                    # (rare) collision / violation events dominate the rare-class
                    # imbalance. Matches SafeDreamer/agent.py:215-218.
                    if name == "cost" and getattr(self._config, "use_cost", False):
                        w = torch.where(
                            data["cost"] >= 1.0,
                            torch.tensor(
                                self._config.cost_weight,
                                device=loss.device,
                                dtype=loss.dtype,
                            ),
                            torch.tensor(1.0, device=loss.device, dtype=loss.dtype),
                        )
                        loss = loss * w
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            # Debug: show per-head loss, KL breakdown, and prediction stats
            if tools.DebugPrinter.every("wm_loss", 100):  # was 20 (round-5)
                tools.DebugPrinter.header("WM-LOSS")
                for k, v in losses.items():
                    tools.DebugPrinter.kv(**{f"loss_{k}": v.mean()})
                # Print discriminates between the *clipped* dyn_loss/rep_loss
                # (which are mathematically equal — the sg/clip only changes
                # gradient flow, not forward value) and the *scaled*
                # contributions that actually enter `total_wm`. Without the
                # scaled view it looks like dyn_loss and rep_loss contribute
                # equally — but they don't (β_dyn=0.5, β_rep=0.1 by default).
                # Disambiguates the cosmetic confusion flagged in the audit.
                _ds = self._config.dyn_scale
                _rs = self._config.rep_scale
                tools.DebugPrinter.kv(
                    kl_unclipped=kl_value.mean(),       # raw KL[post‖prior], no free-bits clip
                    dyn_loss_clipped=dyn_loss.mean(),   # max(KL, free) — gradient → prior
                    rep_loss_clipped=rep_loss.mean(),   # same forward value, gradient → posterior
                    dyn_contrib=(_ds * dyn_loss.mean()),  # β_dyn · dyn_loss inside total_wm
                    rep_contrib=(_rs * rep_loss.mean()),  # β_rep · rep_loss inside total_wm
                    total_wm=model_loss.mean(),
                )
                if "reward" in preds:
                    tools.DebugPrinter.stats(
                        "reward_pred_mode", preds["reward"].mode()
                    )
                    tools.DebugPrinter.stats("reward_target", data["reward"])
                if "cost" in preds:
                    tools.DebugPrinter.stats("cost_pred_mode", preds["cost"].mode())
                    tools.DebugPrinter.stats("cost_target", data["cost"])
                    # Bucket-collapse diagnostic for the symlog_disc(255) cost head.
                    # If `argmax_zero_frac` ≈ 1.0 the head has degenerated to "always
                    # predict cost ≈ 0" — confirms the class-imbalance hypothesis
                    # (cost ≥ 1 events are <1% of replay even with cost_weight=30×).
                    try:
                        cp = preds["cost"]
                        if hasattr(cp, "probs"):
                            flat = cp.probs.detach().reshape(-1, cp.probs.shape[-1])
                            ent = -(flat.clamp_min(1e-8) * flat.clamp_min(1e-8).log()).sum(-1)
                            top = flat.argmax(-1)
                            tools.DebugPrinter.kv(
                                cost_pred_entropy_mean=ent.mean(),
                                cost_pred_entropy_min=ent.min(),
                                cost_pred_argmax_zero_frac=(top == 0).float().mean(),
                                cost_pred_top_bucket_mean=top.float().mean(),
                                cost_pred_top_bucket_max=top.float().max(),
                            )
                    except Exception as e:
                        tools.DebugPrinter.kv(cost_diag_err=repr(e))

            # Deeper look at encoder/decoder/reward head accuracy. Compares
            # reconstructed obs to the actual obs for each numeric key so we
            # can see which parts of the observation the WM is tracking well.
            if tools.DebugPrinter.every("wm_heads_detail", 200):  # was 40 (round-5)
                tools.DebugPrinter.header("WM-HEADS-DETAIL")
                for k in ("ego", "neighbors", "lights", "route_ctx"):
                    if k in preds and k in data:
                        try:
                            recon = preds[k].mode()
                            target = data[k].to(recon.device).to(recon.dtype)
                            abs_err = (recon - target).abs().mean()
                            rel_err = (abs_err / (target.abs().mean() + 1e-6))
                            tools.DebugPrinter.stats(f"recon.{k}", recon)
                            tools.DebugPrinter.stats(f"target.{k}", target)
                            tools.DebugPrinter.kv(
                                **{
                                    f"{k}_abs_err": abs_err,
                                    f"{k}_rel_err": rel_err,
                                }
                            )
                        except Exception as e:
                            tools.DebugPrinter.kv(**{f"{k}_err": repr(e)})
                if "cont" in preds and "cont" in data:
                    try:
                        cp = preds["cont"].mean
                        ct = data["cont"].to(cp.device).to(cp.dtype)
                        tools.DebugPrinter.stats("cont_pred", cp)
                        tools.DebugPrinter.stats("cont_target", ct)
                        tools.DebugPrinter.kv(
                            cont_bce=-(ct * cp.clamp_min(1e-6).log()
                                       + (1 - ct) * (1 - cp).clamp_min(1e-6).log()).mean()
                        )
                    except Exception as e:
                        tools.DebugPrinter.kv(cont_err=repr(e))
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        # Store scalar metrics to avoid keeping (batch,time) arrays until the next log step.
        metrics.update(
            {f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast(
            device_type=str(self._config.device).split(":")[0],
            enabled=self._use_amp,
        ):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        # Image normalisation only when an image channel is present — the SUMO
        # numeric env has no "image" key and should not crash here.
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        # SafeDreamer: cost is a scalar-per-step target for DiscDist head; it
        # stays in the same (B, T) shape as reward (no extra trailing dim).
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        # SafeDreamer cost critic + slow target + own optimizer (mirror of value)
        self._use_cost = bool(getattr(config, "use_cost", False))
        if self._use_cost:
            cc_cfg = config.cost_critic
            self.cost_value = networks.MLP(
                feat_size,
                (255,) if cc_cfg["dist"] == "symlog_disc" else (),
                cc_cfg["layers"],
                config.units,
                config.act,
                config.norm,
                cc_cfg["dist"],
                outscale=cc_cfg["outscale"],
                device=config.device,
                name="CostValue",
            )
            if cc_cfg["slow_target"]:
                self._slow_cost_value = copy.deepcopy(self.cost_value)
                self._cost_updates = 0
            self._cost_value_opt = tools.Optimizer(
                "cost_value",
                self.cost_value.parameters(),
                cc_cfg["lr"],
                cc_cfg["eps"],
                cc_cfg["grad_clip"],
                **kw,
            )
            print(
                f"Optimizer cost_value_opt has {sum(param.numel() for param in self.cost_value.parameters())} variables."
            )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)
            if self._use_cost:
                self.register_buffer(
                    "cost_ema_vals", torch.zeros((2,), device=self._config.device)
                )
                self.cost_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
        lagrange_penalty=0.0,
        effective_cost_limit=None,
    ):
        """Behavior train step.

        Round-5 (2026-04-30) safety knobs:
          - `effective_cost_limit`: when not None, overrides
            `self._config.cost_limit` for the ψ term. Lets the caller
            (Dreamer) tighten the limit adaptively when the world model
            consistently under-predicts real-environment cost.
          - The ψ statistic uses cost_target.quantile(0.85) instead of
            cost_target.mean(), so tail-risk imagined trajectories drive
            the penalty even when the average imagined cost is benign.
        """
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast(
                device_type=str(self._config.device).split(":")[0],
                enabled=self._use_amp,
            ):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                # SafeDreamer cost branch: imagined cost trajectory + cost
                # target for critic + ψ penalty for the actor loss.
                cost_target_list = None
                cost_base = None
                imag_cost = None
                psi = None
                if self._use_cost and "cost" in self._world_model.heads:
                    imag_cost = self._world_model.heads["cost"](imag_feat).mode()
                    cost_target_list, cost_weights, cost_base = self._compute_cost_target(
                        imag_feat, imag_state, imag_cost
                    )
                    # Episode-level constraint violation: E[G^c] - d.
                    # Round-5: the ψ term uses the 85th percentile of cost
                    # targets (not the mean) so tail-risk imagined paths
                    # dominate the penalty signal. The mean was washing
                    # out infrequent-but-large cost events. Using a
                    # quantile is bounded above by max(cost_target), so ψ
                    # has a hard ceiling and can't explode.
                    stacked_cost_target = torch.stack(cost_target_list, dim=1)
                    cl = self._config.cost_limit if effective_cost_limit is None else float(effective_cost_limit)
                    cost_violation = (
                        torch.quantile(stacked_cost_target.flatten(), 0.85)
                        - cl
                    )
                    psi = float(lagrange_penalty) * cost_violation
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                if psi is not None:
                    # ψ is a scalar; adds pressure toward cost ≤ d on average.
                    actor_loss = actor_loss + psi
                    metrics["lagrange_penalty"] = float(lagrange_penalty)
                    metrics["cost_violation"] = to_np(cost_violation.detach())
                    metrics["psi"] = to_np(psi.detach())
                    # Round-5: surface the effective cost limit + the per-
                    # batch imagined cost rate so dreamer.py can update its
                    # imag-vs-real EMA gap tracker.
                    metrics["effective_cost_limit"] = float(cl)
                    metrics["imag_cost_per_step"] = float(imag_cost.mean().item())
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(
                device_type=str(self._config.device).split(":")[0],
                enabled=self._use_amp,
            ):
                value = self.value(value_input[:-1].detach())
                target_stack = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target_stack.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        # Cost critic training (SafeDreamer)
        cost_value_loss = None
        cost_value_mode = None
        if self._use_cost and cost_target_list is not None:
            with tools.RequiresGrad(self.cost_value):
                with torch.amp.autocast(
                    device_type=str(self._config.device).split(":")[0],
                    enabled=self._use_amp,
                ):
                    cv = self.cost_value(value_input[:-1].detach())
                    cost_target_stack = torch.stack(cost_target_list, dim=1)
                    cv_loss = -cv.log_prob(cost_target_stack.detach())
                    slow_cv = self._slow_cost_value(value_input[:-1].detach())
                    if self._config.cost_critic["slow_target"]:
                        cv_loss -= cv.log_prob(slow_cv.mode().detach())
                    cv_loss = torch.mean(cost_weights[:-1] * cv_loss[:, :, None])
                    cost_value_loss = cv_loss
                    cost_value_mode = cv.mode()

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target_stack, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if cost_value_mode is not None:
            metrics.update(tools.tensorstats(cost_value_mode, "cost_value"))
            metrics.update(tools.tensorstats(cost_target_stack, "cost_target"))
            metrics.update(tools.tensorstats(imag_cost, "imag_cost"))
        # Debug: full imagination snapshot every N behaviour-train steps
        if tools.DebugPrinter.every("beh_imag", 100):  # was 20 (round-5)
            tools.DebugPrinter.header("BEH-IMAG", horizon=self._config.imag_horizon)
            tools.DebugPrinter.stats("imag_feat", imag_feat)
            tools.DebugPrinter.stats("imag_action", imag_action)
            tools.DebugPrinter.stats("imag_reward", reward)
            tools.DebugPrinter.stats("reward_target_lambda", target_stack)
            tools.DebugPrinter.stats("weights", weights)
            tools.DebugPrinter.kv(
                actor_loss=actor_loss, value_loss=value_loss, actor_ent=actor_ent.mean()
            )
            if imag_cost is not None:
                tools.DebugPrinter.stats("imag_cost", imag_cost)
                tools.DebugPrinter.stats("cost_target_lambda", cost_target_stack)
                tools.DebugPrinter.kv(
                    lagrange_penalty=float(lagrange_penalty),
                    cost_violation=cost_violation,
                    psi=psi,
                    cost_value_loss=cost_value_loss,
                )
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            if cost_value_loss is not None:
                metrics.update(
                    self._cost_value_opt(cost_value_loss, self.cost_value.parameters())
                )
        return imag_feat, imag_state, imag_action, weights, metrics

    def _compute_cost_target(self, imag_feat, imag_state, imag_cost):
        """Cost TD(λ)-return mirroring _compute_target for rewards. The
        discount comes from the continue head just like the reward path,
        so cost returns truncate correctly when the imagined episode ends.
        """
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(imag_cost)
        value_c = self.cost_value(imag_feat).mode()
        target = tools.lambda_return(
            imag_cost[1:],
            value_c[:-1],
            discount[1:],
            bootstrap=value_c[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value_c[:-1]

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
        # SafeDreamer cost-critic slow target has its own counter + mix
        if self._use_cost and self._config.cost_critic["slow_target"]:
            if self._cost_updates % self._config.cost_critic["slow_target_update"] == 0:
                mix = self._config.cost_critic["slow_target_fraction"]
                for s, d in zip(
                    self.cost_value.parameters(), self._slow_cost_value.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._cost_updates += 1
