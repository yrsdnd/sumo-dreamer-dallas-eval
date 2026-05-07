"""Numeric-only SUMO environment for SafeDreamer-PID training.

One ego vehicle is added into a SUMO scenario via libsumo (in-process,
single-env). Observations are numeric (ego kinematics + N nearest neighbors
+ K upcoming traffic-light states), matching the safedreamer_v3_complete.html
design. Continuous 2-D action: ``[accel_norm, lane_signal]``.

This file is intentionally self-contained: it sets SUMO_HOME and appends
the framework tools directory to sys.path, so no shell export is required
before running ``python3 dreamer.py --configs sumo ...``.

The scenario (subregion netconvert output + random trip rou file) is
expected at ``envs/sumo_scenario/subregion.sumocfg``. A stub generator is
provided so the training code can be wired up end-to-end before we carve
a real subregion from the Dallas OSM city net.
"""
from __future__ import annotations

import math
import os
import pathlib
import random
import sys
from typing import Optional

import numpy as np

import gym

# ---- SUMO framework discovery (macOS Eclipse SUMO install) -----------------
_SUMO_HOME_DEFAULT = (
    "/Library/Frameworks/EclipseSUMO.framework/"
    "Versions/Current/EclipseSUMO/share/sumo"
)
SUMO_HOME = os.environ.get("SUMO_HOME", _SUMO_HOME_DEFAULT)
os.environ["SUMO_HOME"] = SUMO_HOME
_TOOLS = os.path.join(SUMO_HOME, "tools")
if _TOOLS not in sys.path:
    sys.path.append(_TOOLS)

try:  # libsumo first (fast, single-process)
    import libsumo as traci  # type: ignore
    import libsumo.constants as tc  # type: ignore
    _HAVE_LIBSUMO = True
except Exception:
    import traci  # type: ignore
    import traci.constants as tc  # type: ignore
    _HAVE_LIBSUMO = False

# Lazy import: avoid circular on pure-env tests.
def _dbg():
    from tools import DebugPrinter
    return DebugPrinter


# ---- Scenario defaults -----------------------------------------------------
SCENARIO_DIR = pathlib.Path(__file__).parent / "sumo_scenario"
SCENARIO_CFG = SCENARIO_DIR / "subregion.sumocfg"

# Task → cfg mapping. Adding a new task is a 1-line entry. The task name
# comes from config.task after splitting on the first "_" (sumo_subregion
# → "subregion", sumo_dallas_city → "dallas_city"). Any task without an
# explicit mapping falls back to the legacy SCENARIO_CFG (subregion).
_SCENARIO_PATHS = {
    "subregion":   pathlib.Path(__file__).parent / "sumo_scenario" / "subregion.sumocfg",
    "dallas_city": pathlib.Path(__file__).parent / "sumo_city_dallas" / "osm.sumocfg",
}


def _resolve_scenario(task: str) -> pathlib.Path:
    """Map task name → .sumocfg path. Returns SCENARIO_CFG (legacy
    subregion) for unknown task names."""
    return _SCENARIO_PATHS.get(task, SCENARIO_CFG)

EGO_ID = "ego"
EGO_TYPE = "ego_passenger"  # was "veh_passenger" before Phase 2 of Dallas
                            # integration. Keeping a dedicated standalone vType
                            # for ego (defined at top-level in the passenger
                            # trips file) guarantees fixed identity — ego never
                            # randomly inherits dangerous behavior from the
                            # passengerDist vTypeDistribution. The legacy
                            # subregion scenario doesn't define ego_passenger;
                            # SUMO falls back to its DEFAULT_VEHTYPE for that
                            # case (length=5.0 too, so observation stays
                            # consistent — see _spawn_ego_with_retries which
                            # tolerates this fallback gracefully).
STEP_LENGTH = 0.1  # seconds (10 Hz control)

# 2026-05-02 (round-7, queue-based ego insertion).
# Replaces direct `traci.vehicle.add(EGO_ID, ..., depart="now")` with SUMO's
# native vehicle-queue mechanism: ego is added with depart=<future sim time>,
# and the env steps the simulation (no model in the loop) until ego is
# actually inserted by SUMO's safe-gap check. This:
#   - eliminates the manual "find a 25 m gap" retry loop in favor of SUMO's
#     own insertion-safety check (more robust);
#   - lets ego enter at flowing speed like other vehicles, not at rest;
#   - introduces TEMPORAL DIVERSITY into the data distribution — episodes
#     start at different scenario timepoints (early-sparse / mid-busy /
#     late-busy) instead of always at t=30s of the 1-hour scenario.
#
# The empty-RSSM-context concern at random temporal entries is resolved by
# the WM's batch sampling, which already sets is_first=True at every
# 64-step chunk boundary regardless of underlying episode position
# (tools.sample_episodes:662, 674). The WM is therefore explicitly trained
# on "empty GRU at arbitrary scenario time" — this design just makes the
# inference distribution match the training distribution.
#
# Curriculum: t_insert ∈ uniform[t_min, t_max] keyed by self._episode_count.
EGO_INSERT_CURRICULUM = [
    # (max_ep_count, t_insert_min_s, t_insert_max_s)
    # 2026-05-02 (round-7 follow-up #2): thresholds re-scaled to the
    # 300k env-step budget. With ~280 total episodes projected by step
    # 300k (rate slowing from 1/600 → 1/1300 as agent learns), the
    # previous (50, 150, 300) thresholds meant stage 3 (full peak
    # coverage) would NEVER fire. New thresholds put stage 3 in roughly
    # the last half of training.
    #
    # Coverage by % of 300k budget:
    #   eps   1- 25 →  ~5% of run (light only — curriculum intro)
    #   eps  26- 75 → ~20% of run (light → moderate)
    #   eps  76-150 → ~45% of run (light → busy)
    #   eps 151+    → ~55% of run (LAST HALF — full coverage incl. peak)
    (25,    60.0,   300.0),    # eps   1- 25: light only (curriculum-easy intro)
    (75,    60.0,   900.0),    # eps  26- 75: light → moderate
    (150,   60.0,  1800.0),    # eps  76-150: light → busy
    (None,  60.0,  2700.0),    # eps 151+   : FULL coverage 1min → 45min
]
# Hard cap on t_insert. 2700s = 45 min — beyond that the trip schedule
# (which ends at 3600s) has mostly drained and density tapers off. Capping
# at 45 min keeps the late-phase regime still realistic.
INSERT_TIME_MAX = 2700.0

# Queue-insertion safety parameters.
N_QUEUE_RETRIES = 10                  # tries with NEW route at SAME t_insert
QUEUE_TIMEOUT_PER_ATTEMPT = 120.0     # sim seconds to wait for ego to appear
                                       # in vehicle.getIDList() after t_insert.

# 2026-05-02 (round-7 follow-up): scenario-drain safety. The Dallas trip
# file ends at sim_time = 3598.17s; beyond ~3550s most vehicles have
# arrived at their destinations and the city is winding down to empty.
# An ego still acting at sim_time > 3600s would be in a degenerate
# regime (no env-vehicles to interact with). Force-terminate the
# episode at SCENARIO_MAX_SIM_TIME with reason="scenario_drained" so
# the agent never trains on out-of-distribution post-scenario data.
# This ALSO guards against ego acting past the 1-hour scenario boundary
# entirely — a logical-correctness invariant.
SCENARIO_MAX_SIM_TIME = 3550.0

# Legacy constant kept for any external callers; current code uses the
# curriculum function. Value = midpoint of the first curriculum stage.
WARMUP_SECONDS = 180.0
MAX_ACCEL = 2.5  # m/s^2 applied when accel_norm = +1
MAX_DECEL = 5.0  # m/s^2 applied when accel_norm = -1
LANE_CHANGE_COOLDOWN = 4.0  # seconds between lane-change commands (was 1.0;
                            # round-5 2026-04-30 — tighter window to discourage
                            # oscillation; matches Highway-Env / CarDreamer
                            # defaults of 3-5 s for deliberate merges).
NEIGHBOR_RADIUS = 100.0  # meters for context subscription
N_NEIGHBORS = 10
N_TLS_AHEAD = 3
LANE_SIGNAL_THRESHOLD = 0.5  # |lane_signal| ≥ threshold ⇒ request a lane change

# Spawn hardening: require at least this much clear space in front of and behind
# the ego when it is inserted, otherwise reject + retry on a different route.
# SUMO's native insertion check prevents collisions at t=0 but not necessarily
# at t=ε — if our random actor decelerates harder than Krauss expected, a tight
# follower can still rear-end us.
# Tuned 25 → 50 → 30 m during Dallas integration Phase 3 (2026-04-28).
# 50 m proved too restrictive: ego ended up in zero-neighbor dead zones
# even on Dallas's 5896-vehicle city (the spawn would only succeed on
# very quiet streets, defeating the point of training in traffic).
# 30 m strikes the balance — strict enough that no env car materialises
# right on top of ego, loose enough that ego still spawns near some
# traffic. The post-spawn `_remove_close_newcomers` hook (sumo.py)
# adds a second protection layer for vehicles inserted later in the
# episode.
MIN_SPAWN_LEADER_GAP = 30.0   # was 25 originally, bumped to 50, settled at 30
MIN_SPAWN_FOLLOWER_GAP = 30.0
SPAWN_RETRY_ATTEMPTS = 24
EGO_DEPART_SPEED = "5"  # 2026-05-02 (round-7): 5 m/s ≈ 18 km/h — slow but
                         # moving, so a follower at 10-15 m/s can brake safely
                         # within 1-2 sim steps. Was "0" (start at rest), but
                         # rest-spawn is unrealistic and made env-vehicles brake
                         # hard on insertion. A SUMO-native safe-gap check
                         # (see _queue_ego_at) ensures no insertion violates
                         # the follower's deceleration capacity at this speed.

# 2026-05-03 (round-7 follow-up): the M8 blacklist (round-6) was wrong.
# Inspecting osm.net.xml.gz confirmed all 6 originally-blacklisted edges
# AND the round-7 problem edge (441530150#0) have multiple valid out-
# connections. The "emergency stop" events are NOT topology dead-ends —
# they are LANE-MISMATCH issues where ego ends up in a lane without a
# connection to the route's NEXT edge.
#
# Example: edge 441530150#0 (East Mockingbird Lane, secondary highway) has:
#   lane _0 → 90990604#0 (right turn) AND → 825955849 (straight)
#   lane _1 → 825955849 (straight) AND → -91242641#2 (left) AND → 573160086#0 (U-turn)
# If ego is in lane _0 but the route requires the left turn, it can't
# make the maneuver and SUMO emergency-stops at the lane end.
#
# The CORRECT fix is to enable strategic lane changes
# (setLaneChangeMode bit 0 = value 1) so SUMO automatically positions
# ego in the right lane for upcoming maneuvers. See _post_insertion_setup
# below. This obviates the need for the blacklist entirely.
#
# We keep the variable + the filter logic for backward compatibility,
# but it's now an empty set (no-op). To re-blacklist a specific edge
# in the future, just add its ID here.
DEAD_END_EDGES = frozenset()

# 2026-05-03 (round-7 follow-up #4): preferred-lane observation + reward.
# Replaces SUMO's strategic lane-change enforcement (which masked actor
# learning). Ego now has full lane-change autonomy (setLaneChangeMode=0)
# and gets an explicit obs hint about which lane(s) the route requires,
# plus reward shaping to encourage proactive positioning.
#
# Obs key 'preferred_lane' is a 3-dim vector:
#   [0] is_correct_lane     ∈ {0, 1}     — am I in ANY route-correct lane?
#                                          1 if my current lane allows continuation
#                                          along the planned route, 0 otherwise.
#   [1] offset_to_BEST      ∈ [-1, +1]   — signed lane offset to the BEST correct lane.
#                                          BEST = lane with longest continuation
#                                          length (per getBestLanes). When multiple
#                                          lanes tie at max length (e.g., all lanes
#                                          equally good on a straight multi-lane edge),
#                                          we PREFER the current lane → offset=0 means
#                                          "I'm on a max-length lane, no need to move".
#                                          When current lane is wrong or sub-optimal,
#                                          we pick the closest lane (to cur_idx) among
#                                          those tied at max length, minimizing the
#                                          required movement.
#                                          Clamped at ±5 lanes (max in city = 5).
#                                          +0.2 = need 1 lane LEFT, -0.4 = 2 RIGHT.
#   [2] num_correct_frac    ∈ [0, 1]     — fraction of current edge's lanes that
#                                          allow continuation (1.0 = single-lane edge,
#                                          0.4 = 2 of 5 lanes are correct → tight)
#
# Computed via traci.vehicle.getBestLanes() — matches SUMO's internal
# strategic-LC computation EXACTLY (zero risk of mis-encoding).
#
# 2026-05-03 (round-7 #5): refined offset semantics from "nearest correct"
# to "BEST correct". This gives the agent a continuous gradient even when
# already in a correct lane, encouraging movement to the optimal lane
# (longer continuation = fewer downstream constraint satisfactions).
PREFERRED_LANE_DIM = 3

# Reward shaping for the new lane-positioning channel:
R_LANE_CORRECT_BONUS = 0.005           # per-step bonus when in a correct lane
R_TERMINAL_CORRECT_LANE_BONUS = 0.5    # one-shot bonus at arrived-terminal if also in correct lane
R_WRONG_LANE_END_PENALTY = 10.0        # signed magnitude; applied as -10 on wrong_lane_end
                                       # terminal (much smaller than the -50 silent_stop
                                       # penalty — this is a navigation failure, not safety)

# Pending-vehicle deferral buffer: any env vehicle whose insertion would
# happen within this many metres of the ego (on ego's own edge) gets
# removed at the moment it appears (departed-list filter, sumo.py
# `_remove_close_newcomers`). Keeps env-cars from rear-ending or front-
# clipping the ego at insertion time. (Phase 3, 2026-04-28.)
EGO_NO_SPAWN_BUFFER = 30.0   # metres around ego on the same lane

# 2026-05-05 (round-7 #10): spawn guard window. EGO_NO_SPAWN_BUFFER above
# only protects against vehicles that DEPART (newly spawn) close to ego.
# It doesn't protect against:
#   (1) Fast env vehicles already driving that catch up to slow-spawned ego
#       from BEHIND on the same lane → collision_victim (-20, +0.3 cost)
#   (2) Slow env vehicles already in front of ego that ego accelerates
#       INTO (ego rear-ends them) → collision_at_fault (-50, +1.0 cost)
# The spawn-guard window removes ANY env vehicle on ego's lane within
# ±SPAWN_GUARD_RADIUS of ego, for SPAWN_GUARD_STEPS after ego enters.
# After that, ego is on its own and must learn defensive driving.
SPAWN_GUARD_STEPS  = 20      # 2.0 sim-seconds at 10Hz
SPAWN_GUARD_RADIUS = 25.0    # m; vehicles within ±25m on same lane get vaporized

# -- Reward v2 weights (inspired by CarDreamer tasks.yaml: waypoint=2.0,
# speed=0.5, collision=30, out_of_lane=3.0, destination_reached=20).
# Scaled down so per-step reward stays O(0.1–0.5), terminal signals dominate.
R_PROGRESS_SCALE = 0.1       # per metre of route-distance advanced
R_SPEED_SCALE = 0.05         # max per-step speed-tracking reward
R_IDLE_PENALTY = 0.02        # subtracted when stopped without justification
R_ARRIVAL_BONUS = 100.0      # terminal bonus for reaching the destination
R_COLLISION_PENALTY = 50.0   # terminal penalty for any ego collision (base)
# Round-4 reward additions (2026-04-28), inspired by CarDreamer's reward
# decomposition. None of these change the cost (safety) channel; they only
# shape the reward channel so that V_r learns finer behaviour:
#   - R_CTE_SCALE: per-step cross-track-error penalty when ego drifts off
#     the lane centre. Zero inside the central half (deadzone w/4); ramps
#     linearly to -R_CTE_SCALE at the lane edge.
#   - R_HEADING_SCALE: per-step heading-vs-lane-direction penalty. Uses
#     |sin(Δθ)| where Δθ is the angle between ego heading and the current
#     lane's forward direction (already computed for route_ctx[4]).
#   - R_COLLISION_SPEED_BONUS: extra collision penalty scaled by ego speed
#     at impact. A standstill-bump still costs R_COLLISION_PENALTY=50; a
#     full-speed crash costs +R_COLLISION_SPEED_BONUS more on top, giving
#     the actor a stronger gradient against high-energy approaches.
#
# 2026-05-02 (M3 fix): R_COLLISION_SPEED_BONUS reduced 50 → 0 to symmetrize
# the reward distribution. The 50+50=100 floor for at-fault collisions made
# value targets asymmetric vs the +60 best-arrival ceiling, slowing value
# learning. Speed-scaled safety signal is now carried by the cost channel
# (rule_violation_cost + collision flag) — that's the right place for it.
R_CTE_SCALE = 0.05
R_HEADING_SCALE = 0.02
R_COLLISION_SPEED_BONUS = 0.0  # was 50.0 (M3 fix 2026-05-02)

# Round-5 (2026-04-30) — absolute speed penalties (per-step, additive to reward).
# Earlier rounds only penalised speeding via the cost channel and only when
# the speed-context was "unsafe" (leader / TLS / junction nearby). On clearly
# empty roads, speeding had ZERO cost, so the agent learned to floor it on
# empty stretches and reach 50-80 m/s without consequence. New rule:
#   - Speed > 1.15 × lane.maxSpeed: small per-step reward penalty regardless
#     of context. Encodes "you're 15% over limit, slow down" universally.
#   - Speed > ABS_SPEED_CAP (= 30 m/s ≈ 108 km/h): small penalty regardless
#     of speed-limit. Encodes a hard "no urban driving above 30 m/s" cap so
#     even on rare unrealistic-limit edges the agent doesn't fly.
# Both penalties are SMALL (per-step ≤ R_PROGRESS_SCALE × few_metres) so they
# shape behaviour without dominating the reward signal. The ψ-channel cost
# upgrade (also round-5) is more aggressive — these are the soft analogue.
LIMIT_OVERAGE_THRESHOLD = 0.15        # speed > 1.15 × lane vmax → penalty
ABS_SPEED_CAP = 30.0                  # m/s, absolute cap regardless of limit
R_LIMIT_OVERAGE_PENALTY = 0.05        # max per-step penalty for limit overage
R_ABS_SPEED_PENALTY     = 0.05        # max per-step penalty for absolute cap

# Round-5 reward additions (2026-04-30) — lane-change discipline + collision
# fault differentiation + arrival-bonus curriculum.
#
# Lane-change discipline: the prior run (logs/dallas_long, ~33k steps) showed
# 80% of action emissions had |lane_signal| > 0.5 but 94% of those were
# silent no-ops (already on edge lane / 1-lane edge / inside cooldown). The
# lane_signal channel had near-zero gradient signal because no-op actions
# produce no consequence. Result: actor's lane_signal output drifted to a
# corner. Fix: small per-step penalty for futile commands gives the channel
# real gradient. Oscillation penalty (commanding opposite direction within
# 6 s of last successful merge) discourages thrashing.
R_FUTILE_MERGE_PENALTY      = 0.01    # |lane_signal|>0.5 but no fire
R_MERGE_OSCILLATION_PENALTY = 0.05    # opposite-direction cmd within window
MERGE_OSCILLATION_WINDOW    = 6.0     # seconds since last fire

# Collision fault differentiation: SUMO's collision struct exposes `collider`
# (vehicle that initiated the impact, typically rear) and `victim` (vehicle
# hit, typically front). Prior code OR'd both as "ego_collided" with full
# penalty regardless of fault. With the driver-diversity changes, dangerous
# env-cars sometimes rear-end the ego — penalising that the same as ego
# rear-ending someone is incorrect. Round-5: 40% of full penalty when ego
# is the victim (still terminates the episode, but agent isn't catastrophi-
# cally punished for an unavoidable hit).
R_VICTIM_COLLISION_RATIO = 0.4        # 0.4 × full collision reward penalty
COST_VICTIM_COLLISION    = 0.3        # cost when ego is the victim, not at fault

# Arrival-bonus scale (variant d): fixed base + per-metre bonus. Easy 1500m
# route → +62 bonus; hard 5000m → +150. Computed at spawn from the picked
# route's length and stored on self for the terminal-arrival branch to read.
# Replaces the constant R_ARRIVAL_BONUS=100 → length-aware shaping so harder
# routes are proportionally more rewarding without devaluing easy success.
R_ARRIVAL_BONUS_BASE     = 25.0       # baseline regardless of length
R_ARRIVAL_BONUS_PER_M    = 0.025      # +1 bonus per 40 m of route length

# Route-length curriculum (4 stages, scaled by self._episode_count).
# Goal: the agent has to experience arrivals to learn navigation. Long
# Dallas routes (avg ~3.5 km) couldn't be completed in the prior 3000-step
# / 5-min budget; result was 0 arrivals in 30 episodes. Curriculum starts
# with short routes that fit easily in the new 6000-step budget, gradually
# expands. Tracked by self._episode_count (incremented each reset).
CURRICULUM_STAGES = [
    # (max_episode_count_for_stage, max_route_length_m)
    (50,  1500.0),    # eps 1-50: very short — learn basic arrival + navigation
    (150, 2500.0),    # eps 51-150: medium
    (300, 4000.0),    # eps 151-300: longer city routes
    (None, float("inf")),  # eps 301+: any length
]


# Observation normalization (added 2026-04-27).
# Bug: raw distance observations (lights TLS dist, route_ctx dist_to_route_end)
# were unbounded — values up to 2.2 km appeared in the smoke trace. After the
# decoder's SymlogDist computes (mode - symlog(2211))² ≈ 59 per dim, the loss
# for these keys dominated L_wm (lights+route_ctx contributed 65-85% of total),
# the first gradient norm exceeded the clip ceiling (1170 > 1000), and the WM
# loss trajectory was bumpy (steps 6-7 spiked from 56 → 86 → 115). The actor
# also collapsed to brake-only because reward-path gradients were dwarfed by
# decoder gradients on the unbounded keys.
#
# Fix: divide every distance / speed / size observation by a configured maximum
# before returning. All obs dims now sit roughly in [-1, 1], so the encoder /
# decoder operate on consistently-scaled inputs and gradients stay balanced.
# NOTE: this changes the obs distribution; old checkpoints (logs/safe_pid_long)
# will not work — re-training is required for runs that depend on the new env.
MAX_OBS_DIST = 200.0    # m;   distances clipped to this then divided by it
MAX_OBS_SPEED = 30.0    # m/s; ~108 km/h covers SUMO highway max speeds
MAX_OBS_ACCEL = 10.0    # m/s²; ego accel rarely exceeds ±5 m/s², headroom for spikes
MAX_OBS_LANE = 5.0      # used for lane_idx, num_lanes_here, and lane_delta
MAX_OBS_LEN = 10.0      # m;   vehicle length & lane width divided by this
# Closure rate between ego and any neighbor can be up to 2·vmax (head-on at the
# speed limit on a two-way street). Dividing by 2·MAX_OBS_SPEED keeps rel_v in
# [-1, 1]. Without this, rel_v_long touched -1.24 in the 800-step probe — the
# encoder's input range assumption is violated. (F-OBS-1, 2026-04-27.)
MAX_REL_V = 2.0 * MAX_OBS_SPEED

# Silent-failure detection (added 2026-04-27 after the
# tests/test_collision_dream_vs_real.py investigation observed an ego whose
# speed dropped from 74 m/s → 0 m/s in a single env step yet was reported as
# done=False, c=0 by the existing branches — neither getCollisions() fired
# nor was the ego removed from getIDList(). The most likely cause is a SUMO
# event that warns but doesn't surface through the collision API (junction
# internal-edge clamp, soft jam, etc.). Either way the agent shouldn't be
# allowed to silently sit there earning idle penalties forever, and this
# class of failure should be fed into the cost head so the actor learns to
# avoid it.
SUDDEN_STOP_PRIOR_SPEED = 5.0    # m/s; if last_speed > this AND ...
SUDDEN_STOP_CUR_SPEED = 0.1      #      ... cur_speed below this with no
                                 #          justification → treat as silent failure
# Bumped 50 → 100 (5 s → 10 s) on 2026-04-27 because the brief-train audit
# showed 95% of random-prefill episodes terminating at exactly step 50: the
# uniform Box(-1,1) prefill actor has E[accel_norm]=0, so ego barely moves
# and tripped the stuck detector before any real failure mode could be
# observed. 10 s is still well below human "this car is broken" reaction
# time and continues to catch genuine gridlock / dead-end stops.
STUCK_THRESHOLD_STEPS = 100      # 10 s at 10 Hz; consecutive unjustified-idle
                                 # steps before declaring "stuck"

# Feature dim layout (added 2026-04-27, then expanded):
#   ego[9]       = road_type     ∈ [0, 1] (0=junction, 1=highway/primary)
#   ego[10]      = ego.length    ∈ [0, 1] (length / MAX_OBS_LEN)         (F-OBS-3)
#   ego[11]      = is_at_left_edge   ∈ {0, 1} (round-5, 2026-04-30)
#   ego[12]      = is_at_right_edge  ∈ {0, 1} (round-5, 2026-04-30)
#   neighbor[8]  = lane_delta    ∈ [-1, +1] (nb.lane_idx − ego.lane_idx, /MAX_OBS_LANE)  (F-OBS-6)
#   neighbor[9]  = signal        ∈ {-1, 0, +1} (right blinker = -1, left = +1) (F-OBS-4)
# Rationale: lane-structure + ego self-extent + neighbor merge intent without
# breaking the lane-invariant ego-relative frame.
EGO_FEAT_DIM = 13       # was 11; +2 for is_at_left_edge / is_at_right_edge
                        # (round-5 2026-04-30) — direct observable booleans
                        # for "I'm at a lane edge" so the actor's lane_signal
                        # head learns to suppress futile merges quickly.
NEIGHBOR_FEAT_DIM = 10
TL_FEAT_DIM = 5
ROUTE_CTX_DIM = 6


class SumoEnv(gym.Env):
    """Gym-compatible wrapper around a SUMO TraCI session, with a
    **lane-invariant (Frenet-in-ego-frame)** observation so the RSSM does not
    see discrete jumps when the ego changes lane. Design rationale is in the
    "State redesign" note in the project plan.

    Observation keys (all numeric, all float32 except the two bool flags):
      * ``ego`` — (13,)
          [speed, accel_est, heading_sin, heading_cos,
           lane_index, num_lanes_here, lane_pos_norm,
           lane_speed_limit, lane_width, road_type, ego_length,
           is_at_left_edge, is_at_right_edge]
          The two `is_at_*_edge` booleans (round-5 2026-04-30) make the
          "lane-merge in this direction is futile" condition trivially
          observable. Without them, the encoder has to learn the equality
          `lane_idx == num_lanes - 1` from continuous-valued inputs,
          which is slow. With them, the actor's lane_signal head pairs
          directly with R_FUTILE_MERGE_PENALTY for fast learning.
      * ``neighbors`` — (N_NEIGHBORS × 10,) ranked by Manhattan-Frenet distance,
          padded with presence=0 / class_id=-1. Each slot:
          [presence, long_off, lat_off, rel_v_long, rel_v_lat,
           heading_diff_sin, length, class_id, lane_delta, signal]
          long_off/lat_off are (dx,dy) rotated into the ego heading frame
          (positive long_off = ahead, positive lat_off = to ego's left).
          rel_v_*  divided by 2·MAX_OBS_SPEED so closure rates stay in [-1, 1].
          lane_delta ∈ [-1, 1] (nb.lane − ego.lane, divided by MAX_OBS_LANE).
          signal: +1 left blinker, -1 right blinker, 0 none/both.
      * ``lights`` — (N_TLS_AHEAD × 5,) [dist, red, yellow, green, time_placeholder]
      * ``route_ctx`` — (6,)
          [dist_to_next_junction, dist_to_lane_count_change,
           upcoming_lane_count_delta, dist_to_route_end,
           heading_offset_from_route_sin, target_lane_delta]
          dist_to_lane_count_change = -1 if none within horizon.
          upcoming_lane_count_delta < 0 = merge ahead, > 0 = split ahead.
      * ``is_first``, ``is_terminal``, ``cost`` — as before.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "subregion",
        seed: int = 0,
        cfg_path: Optional[str] = None,
        gui: bool = False,
        gui_delay_ms: int = 100,
        gui_track_ego: bool = True,
    ):
        self._task = task
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        # cfg_path explicit override > task-derived path > legacy default
        if cfg_path:
            self._cfg = pathlib.Path(cfg_path)
        else:
            self._cfg = _resolve_scenario(task)
        self._step_idx = 0
        self._last_lane_change_t = -1e9
        self._last_speed = 0.0
        self._stuck_steps = 0          # consecutive unjustified-idle steps
        self._running = False
        self._prev_route_dist = 0.0   # cumulative along-route distance at last step
        self._spawn_idx = 0           # for lane-count rotation in _pick_route (F-OBS-5)
        # 2026-05-05 (round-7 #10): spawn-guard counter — set in _post_insertion_setup.
        self._spawn_guard_steps_remaining = 0
        # 2026-05-02 (round-7): per-episode trackers used by [DBG EPISODE-SUMMARY].
        self._ep_speed_sum = 0.0
        self._ep_speed_n = 0
        self._ep_reward_sum = 0.0
        self._ep_cost_sum = 0.0
        self._ep_t_insert = 0.0
        # 2026-05-03 (round-7 #4): merge-channel telemetry per episode.
        # _ep_merge_cmds counts steps where |lane_signal|>0.5 (commanded change).
        # _ep_real_lane_changes counts actual lane-id changes between consecutive
        # steps. Ratio actual/cmd reveals "futile spam" rate.
        self._ep_merge_cmds = 0
        self._ep_real_lane_changes = 0
        self._ep_prev_lane_idx = -1
        # Recent-route memory (Dallas long-train round, 2026-04-28).
        # Avoids picking the same (from, to) edge pair too often during long
        # training: keeps a sliding window of the last K=200 route pairs and
        # rejects any new pick that's already in the window. With 1728 edges
        # → ~3M possible pairs, K=200 only excludes a tiny tail.
        from collections import deque
        self._recent_routes = deque(maxlen=200)
        # Round-5 (2026-04-30): episode counter for curriculum stages, and
        # last-merge tracking for lane-change discipline. _last_action_meta
        # is populated by _apply_action and consumed by _reward_v2_shaped.
        self._episode_count = 0
        self._last_merge_dir = 0   # +1 = went left, -1 = went right
        self._last_action_meta = {
            "fired": False,
            "futile_cmd": False,
            "oscillation": False,
        }
        # Arrival bonus computed once at spawn from the actual route length;
        # consumed at terminal-arrival branch. Replaces constant R_ARRIVAL_BONUS.
        self._arrival_bonus_for_episode = float(R_ARRIVAL_BONUS_BASE)
        # GUI options: start sumo-gui instead of headless sumo, slow playback,
        # auto-centre + track ego vehicle with a bright colour.
        self._gui = bool(gui)
        self._gui_delay_ms = int(gui_delay_ms)
        self._gui_track_ego = bool(gui_track_ego)

        nb_dim = N_NEIGHBORS * NEIGHBOR_FEAT_DIM
        tl_dim = N_TLS_AHEAD * TL_FEAT_DIM
        self.observation_space = gym.spaces.Dict(
            {
                "ego": gym.spaces.Box(-np.inf, np.inf, (EGO_FEAT_DIM,), np.float32),
                "neighbors": gym.spaces.Box(-np.inf, np.inf, (nb_dim,), np.float32),
                "lights": gym.spaces.Box(-np.inf, np.inf, (tl_dim,), np.float32),
                "route_ctx": gym.spaces.Box(-np.inf, np.inf, (ROUTE_CTX_DIM,), np.float32),
                # 2026-05-03 (round-7 #4): preferred-lane info for the actor.
                # See PREFERRED_LANE_DIM constant for semantics.
                "preferred_lane": gym.spaces.Box(-1.0, 1.0, (PREFERRED_LANE_DIM,), np.float32),
                "is_first": gym.spaces.Box(0, 1, (), bool),
                "is_terminal": gym.spaces.Box(0, 1, (), bool),
                "cost": gym.spaces.Box(0.0, 1e6, (), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.reward_range = (-np.inf, np.inf)

    # ----------------------- gym API -------------------------------------
    def reset(self):
        self._close()
        if not self._cfg.exists():
            raise FileNotFoundError(
                f"SUMO scenario not found at {self._cfg}. "
                "Generate one with envs/sumo_scenario/build_subregion.py."
            )
        bin_dir = pathlib.Path(SUMO_HOME).parent.parent / "bin"
        sumo_bin = str(bin_dir / ("sumo-gui" if self._gui else "sumo"))
        cmd = [
            sumo_bin,
            "-c",
            str(self._cfg),
            "--step-length",
            str(STEP_LENGTH),
            "--collision.action",
            # Round-5 (2026-04-30): "warn" let SUMO continue post-collision
            # without physically separating vehicles, which on rare occasions
            # could let ego "phase through" traffic. "teleport" forces both
            # vehicles to a teleport zone for a few steps after detection,
            # so collisions are unambiguous. getCollisions() still reports
            # the event so our terminal handler still fires.
            "teleport",
            # Report collisions that happen inside junction internal lanes too
            # (crossing traffic conflicts). Without this, ego could be T-boned
            # at an intersection and the event would be silently ignored.
            "--collision.check-junctions",
            "true",
            "--time-to-teleport",
            "-1",
            # Round-5: removed --no-warnings flag so SUMO collision warnings
            # appear in stderr. We previously suppressed them which made post-
            # hoc verification of getCollisions() impossible. Tradeoff: a bit
            # more locale/proj.db spam in the log; harmless.
            "--no-step-log",
            "--seed",
            str(self._rng.randint(0, 2**31 - 1)),
        ]
        if self._gui:
            # Auto-start + do not quit on end so the user can rewind. Delay
            # is in milliseconds per step → controls playback speed.
            cmd += [
                "--start", "true",
                "--quit-on-end", "false",
                "--delay", str(self._gui_delay_ms),
            ]
        traci.start(cmd)
        self._running = True
        # 2026-05-02 (round-7): queue-based ego insertion replaces the
        # warmup-then-direct-spawn pattern. Flow:
        #   1. Pick a target sim-time t_insert from EGO_INSERT_CURRICULUM.
        #   2. _queue_ego_at(t_insert) adds ego with depart=t_insert and
        #      a fresh route id; SUMO defers insertion until that time.
        #   3. _wait_for_insertion steps the simulation (sim-only, no
        #      model in the loop) until ego appears in getIDList() or
        #      we timeout.
        #   4. On timeout, retry with a NEW route at the SAME t_insert.
        #   5. After N_QUEUE_RETRIES exhausted: hard-fail with diagnostic.
        # This guarantees ego enters at a natural insertion point with a
        # SUMO-verified safe gap, just like every other env vehicle.
        self._episode_count += 1
        t_insert = self._select_ego_insertion_time()
        if _dbg().enabled and (self._episode_count % 10 == 0):
            _dbg().header(
                "INSERT-PLAN",
                episode=self._episode_count,
                t_insert=round(t_insert, 1),
            )
        retry_diagnostics = []
        inserted = False
        # 2026-05-05 (round-7 #9 F3): tight-spawn retry budget — if the chosen
        # route puts ego on a non-redundant lane (frac<1.0), throw it out and
        # re-pick a different route, up to TIGHT_RETRY_BUDGET times. After
        # the budget is exhausted, accept any tight spawn (some scenarios
        # genuinely have only tight options).
        TIGHT_RETRY_BUDGET = 3
        tight_rejections = 0
        for retry in range(N_QUEUE_RETRIES):
            try:
                from_edge, to_edge, edge_list = self._queue_ego_at(t_insert)
            except Exception as e:
                retry_diagnostics.append(
                    f"retry={retry}: route-pick failed: {e!r}"
                )
                continue
            # F3: check spawn-tightness BEFORE committing to insertion. If
            # this is a tight spawn (frac<1.0) AND we still have tight-retry
            # budget AND we still have plenty of overall retries, throw out
            # this candidate and try a different route.
            choice = getattr(self, "_last_spawn_choice", None) or {}
            spawn_frac = float(choice.get("frac", 1.0))
            retries_left = N_QUEUE_RETRIES - retry - 1
            if (spawn_frac < 0.999 and tight_rejections < TIGHT_RETRY_BUDGET
                    and retries_left >= 3):
                tight_rejections += 1
                _dbg().header(
                    "SPAWN-TIGHT-REJECT",
                    retry=retry,
                    tight_rejections=tight_rejections,
                    from_edge=from_edge,
                    frac=round(spawn_frac, 3),
                    note=f"frac<1.0 — re-picking route (budget left: "
                         f"{TIGHT_RETRY_BUDGET - tight_rejections})",
                )
                # Clean up the queued (but not yet inserted) ego before retry
                try:
                    traci.vehicle.remove(EGO_ID, reason=tc.REMOVE_VAPORIZED)
                except Exception:
                    pass
                continue
            inserted = self._wait_for_insertion(t_insert)
            if inserted:
                self._post_insertion_setup(t_insert)
                break
            # Failed: clean up the pending ego entry before next retry.
            try:
                traci.vehicle.remove(EGO_ID, reason=tc.REMOVE_VAPORIZED)
            except Exception:
                pass
            retry_diagnostics.append(
                f"retry={retry}: t_insert={t_insert:.1f} from={from_edge} "
                f"to={to_edge} sim_time={traci.simulation.getTime():.1f} "
                f"n_vehicles={len(traci.vehicle.getIDList())}"
            )
            _dbg().header(
                "QUEUE-RETRY",
                retry=retry,
                t_insert=round(t_insert, 1),
                from_edge=from_edge,
                to_edge=to_edge,
            )
        if not inserted:
            diag = "\n  ".join(retry_diagnostics)
            raise RuntimeError(
                f"FATAL: ego could not be inserted after {N_QUEUE_RETRIES} "
                f"retries at t_insert={t_insert:.1f}s.\n"
                f"This indicates SUMO scenario is not configured to give "
                f"ego priority in same-edge departures, OR every retry's "
                f"chosen route is congested at t_insert. Check trip "
                f"insertion priority + scenario density.\n"
                f"Episode: {self._episode_count}\n"
                f"Retry log:\n  {diag}"
            )
        self._step_idx = 0
        self._last_lane_change_t = -1e9
        self._last_speed = 0.0
        self._stuck_steps = 0
        # 2026-05-02 (round-7): per-episode tracking for [DBG EPISODE-SUMMARY].
        self._ep_speed_sum = 0.0
        self._ep_speed_n = 0
        self._ep_reward_sum = 0.0
        self._ep_cost_sum = 0.0
        self._ep_t_insert = float(t_insert)
        # 2026-05-03 (round-7 #4): merge channel telemetry per episode.
        self._ep_merge_cmds = 0
        self._ep_real_lane_changes = 0
        try:
            self._ep_prev_lane_idx = int(traci.vehicle.getLaneIndex(EGO_ID))
        except Exception:
            self._ep_prev_lane_idx = -1
        # Round-5: reset merge-discipline tracking each episode.
        self._last_merge_dir = 0
        self._last_action_meta = {
            "fired": False,
            "futile_cmd": False,
            "oscillation": False,
            "lane_signal_value": 0.0,
        }
        # Progress baseline: 0 at the moment of spawn. First step's progress
        # will be the first chunk of route distance covered.
        try:
            self._prev_route_dist = float(traci.vehicle.getDistance(EGO_ID))
        except Exception:
            self._prev_route_dist = 0.0
        obs = self._observe(is_first=True, is_terminal=False, cost=0.0)
        # 2026-05-05 (round-7 #9 F6): per-episode start telemetry — dumps
        # the initial obs so post-hoc forensic analysis can correlate
        # spawn-state with episode outcome. Only the most informative
        # subset to keep log size sane.
        try:
            choice = getattr(self, "_last_spawn_choice", None) or {}
            pl = obs.get("preferred_lane")
            ego = obs.get("ego")
            rc = obs.get("route_ctx")
            _dbg().header(
                "EPISODE-START",
                episode=self._episode_count,
                t_insert=round(t_insert, 1),
                from_edge=choice.get("from_edge", "?"),
                spawn_lane_idx=choice.get("chosen_idx", "?"),
                correct_lane_idxs=choice.get("correct_lane_idxs",
                                             choice.get("correct_idxs", "?")),
                n_lanes=choice.get("n_lanes", "?"),
                spawn_frac=round(float(choice.get("frac", 0.0)), 3),
                obs_is_correct=round(float(pl[0]), 2) if pl is not None else "?",
                obs_offset=round(float(pl[1]), 3) if pl is not None else "?",
                obs_n_correct_frac=round(float(pl[2]), 3) if pl is not None else "?",
                ego_speed=round(float(ego[0]) if ego is not None else 0.0, 3),
                ego_lane_idx=round(float(ego[4]) if ego is not None else 0.0, 3),
                ego_num_lanes=round(float(ego[5]) if ego is not None else 0.0, 3),
            )
        except Exception:
            pass
        return obs

    def step(self, action):
        assert self._running, "env.step() called before reset()"
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        accel_norm = float(np.clip(a[0], -1.0, 1.0))
        lane_signal = float(np.clip(a[1], -1.0, 1.0))
        self._apply_action(accel_norm, lane_signal)
        traci.simulationStep()
        self._step_idx += 1
        # Phase 3 (Dallas integration, 2026-04-28): immediately after each
        # sim step, remove any env vehicle that just inserted within
        # EGO_NO_SPAWN_BUFFER of ego on the same lane. SUMO's insertion
        # safety prevents geometric overlap but does not enforce a time-
        # headway buffer — this hook does. Keeps the ego's training cost
        # signal clean from "vehicle materialised in front of me" events.
        self._remove_close_newcomers()
        # 2026-05-05 (round-7 #10): spawn-guard window. For the first
        # SPAWN_GUARD_STEPS (=20 = 2 sim-seconds) after insertion, also
        # vaporize any env vehicle already on ego's lane within ±25m.
        # Eliminates "rear-ended at spawn" and "spawn-into-trap" failures.
        self._spawn_guard_clear_neighbors()

        done = False
        cost = 0.0
        reward = 0.0
        terminated_by_collision = False
        terminated_by_arrival = False
        # Debug: periodic env snapshot
        # Round-5: was 100 → 300 to reduce log spam during long runs.
        if _dbg().every("env_step", 300):
            ego_alive_dbg = EGO_ID in traci.vehicle.getIDList()
            speed_dbg = (
                traci.vehicle.getSpeed(EGO_ID) if ego_alive_dbg else float("nan")
            )
            cur_t = round(traci.simulation.getTime(), 2)
            _dbg().header(
                "ENV",
                step=self._step_idx,
                t=cur_t,
                ego_alive=ego_alive_dbg,
            )
            _dbg().kv(
                accel_norm=accel_norm,
                lane_signal=lane_signal,
                speed=speed_dbg,
                n_vehicles=len(traci.vehicle.getIDList()),
            )
            # 2026-05-02 (round-7): warn when sim_time approaches the
            # scenario boundary so the operator notices before drain
            # termination kicks in (helps diagnose late-curriculum cases).
            if cur_t > 3000.0:
                _dbg().header(
                    "SCENARIO-DRAIN-WARN",
                    sim_time=cur_t,
                    until_drain=round(SCENARIO_MAX_SIM_TIME - cur_t, 1),
                )
        # 2026-05-02 (round-7): scenario-drain safety. If ego is still
        # active after the scenario has effectively ended, force-terminate
        # the episode so we never train on out-of-distribution data
        # (city draining, no env-vehicles to interact with). This is also
        # a logical-correctness guard: ego must not act past the 1-hour
        # simulation boundary.
        sim_time_now = float(traci.simulation.getTime())
        if sim_time_now >= SCENARIO_MAX_SIM_TIME:
            cost = 0.0
            reward = 0.0  # neutral terminal — agent didn't fail, scenario just ended
            done = True
            obs = self._observe(is_first=False, is_terminal=True, cost=0.0)
            _dbg().header(
                "SCENARIO-DRAIN",
                step=self._step_idx,
                sim_time=round(sim_time_now, 1),
                limit=SCENARIO_MAX_SIM_TIME,
            )
            _dbg().header(
                "TERMINAL",
                step=self._step_idx,
                reason="scenario_drained",
                reward=round(reward, 3),
                cost=round(cost, 3),
            )
            info = {"step_idx": self._step_idx, "term_reason": "scenario_drained"}
            return obs, float(reward), bool(done), info

        # Detect hard events before trying to observe
        ego_alive = EGO_ID in traci.vehicle.getIDList()
        collisions = traci.simulation.getCollisions()
        # Round-5: differentiate fault. SUMO's collision struct names the
        # initiator as `collider` (typically the rear/lateral attacker) and
        # the hit vehicle as `victim`. Treating "ego is victim" the same as
        # "ego is collider" caused over-punishment when aggressive env-cars
        # rear-ended ego in the prior run; this branch applies a smaller
        # penalty in the victim case while still terminating the episode.
        ego_at_fault = any(
            getattr(c, "collider", "") == EGO_ID for c in collisions
        )
        ego_was_hit = any(
            getattr(c, "victim", "") == EGO_ID for c in collisions
        )
        ego_collided = ego_at_fault or ego_was_hit
        term_reason = None
        if ego_collided:
            speed_factor = self._collision_speed_factor()
            base_pen = R_COLLISION_PENALTY + R_COLLISION_SPEED_BONUS * speed_factor
            if ego_at_fault:
                cost += 1.0
                reward -= base_pen
                term_reason = "collision_at_fault"
            else:
                # Ego was rear-ended / sideswiped by another driver. Still
                # ends the episode (env vehicle removal makes obs unstable),
                # but agent was not the initiator → reduced penalty + cost.
                cost += COST_VICTIM_COLLISION
                reward -= base_pen * R_VICTIM_COLLISION_RATIO
                term_reason = "collision_victim"
            done = True
            terminated_by_collision = True
        elif not ego_alive:
            arrived_list = set(traci.simulation.getArrivedIDList())
            loaded_list = set(traci.simulation.getLoadedIDList())
            # Some SUMO versions don't repopulate arrivedIDList consistently
            # for vehicles whose route ends inside the subregion's boundary;
            # treat "removed, not collided, not still loaded" as arrival too.
            if EGO_ID in arrived_list or EGO_ID not in loaded_list:
                # Round-5: per-route bonus = base + per_m × route_length_m,
                # set in _pick_route. Replaces the constant R_ARRIVAL_BONUS
                # so easier short routes don't out-reward harder long ones.
                reward += float(self._arrival_bonus_for_episode)
                # 2026-05-03 (round-7 #4): bonus for arriving in a correct lane.
                # Encourages "end with proper lane positioning" (relevant for the
                # route's final edge — typically all lanes are correct here, but
                # the signal still trains the head consistently).
                try:
                    is_correct_at_end, _, _ = self._compute_preferred_lane_info()
                    if is_correct_at_end > 0.5:
                        reward += R_TERMINAL_CORRECT_LANE_BONUS
                except Exception:
                    pass
                done = True
                terminated_by_arrival = True
                term_reason = (
                    "arrived" if EGO_ID in arrived_list else "route_end"
                )
            else:
                cost += 1.0
                reward -= R_COLLISION_PENALTY + R_COLLISION_SPEED_BONUS * self._collision_speed_factor()
                done = True
                term_reason = "unsafe_removal"
        else:
            # Ego still in vehicle list and no SUMO-reported collision —
            # but check for SILENT FAILURES the existing branches miss:
            # (a) sudden speed drop (ego goes from fast to ~0 with no
            #     justification — likely an unreported event), or
            # (b) prolonged unjustified idle (gridlock / dead-end stop).
            #
            # 2026-05-03 (round-7 #8): FAST-PATH wrong_lane_end detection.
            # When ego is force-stopped at the end of a wrong-lane (SUMO
            # clamps the speed via network-topology constraint because no
            # downstream connection exists from this lane), fire the terminal
            # IMMEDIATELY rather than waiting for the 10-second stuck timer.
            # Without this fast-path the silent_failure→stuck branch would
            # spam ~100 emergency-stop warnings per occurrence and waste 10s
            # of simulation while the agent gets no signal. With it, the
            # actor receives the −10 wrong_lane_end penalty within 1 step
            # of SUMO clamping the speed, sharpening credit assignment for
            # whichever earlier lane decision led here.
            #
            # Trigger: speed < 1.0 m/s AND _detect_wrong_lane_end()=True.
            # _detect_wrong_lane_end() already requires ego within 5m of
            # the lane end with no link to route's next edge — there is no
            # recovery possible from this state, so we don't need to wait.
            try:
                _fp_speed = float(traci.vehicle.getSpeed(EGO_ID))
            except Exception:
                _fp_speed = 0.0
            fast_path_wle = (_fp_speed < 1.0 and self._detect_wrong_lane_end())

            silent_done, silent_reason = (False, None)
            if not fast_path_wle:
                silent_done, silent_reason = self._detect_silent_failure()
            slow_path_wle = (silent_done and self._detect_wrong_lane_end())

            if fast_path_wle or slow_path_wle:
                # 2026-05-03 (round-7 #4): WRONG_LANE_END — navigation failure,
                # not safety failure. Reduced penalty (-10 vs -50) and cost=0
                # (no rule violation, just bad routing). The agent was already
                # losing the per-step lane bonus during the approach; we don't
                # double-charge for the failure.
                cost = 0.0
                reward -= R_WRONG_LANE_END_PENALTY
                done = True
                term_reason = "wrong_lane_end"
                # Detailed diagnostic for debugging.
                try:
                    _wlane_id = traci.vehicle.getLaneID(EGO_ID)
                    _wlane_pos = float(traci.vehicle.getLanePosition(EGO_ID))
                    _wlane_len = float(traci.lane.getLength(_wlane_id))
                    _wedge = traci.vehicle.getRoadID(EGO_ID)
                    _wroute = traci.vehicle.getRoute(EGO_ID)
                    _wridx = traci.vehicle.getRouteIndex(EGO_ID)
                    _wnext = _wroute[_wridx + 1] if _wridx + 1 < len(_wroute) else "(none)"
                    _wlinks = [l[0] for l in traci.lane.getLinks(_wlane_id)]
                except Exception:
                    _wlane_id = _wedge = _wnext = "?"
                    _wlane_pos = _wlane_len = -1.0
                    _wlinks = []
                _dbg().header(
                    "WRONG-LANE-TERMINAL",
                    path=("fast" if fast_path_wle else "slow"),
                    speed=round(_fp_speed, 2),
                    lane=_wlane_id,
                    edge=_wedge,
                    lane_pos=round(_wlane_pos, 1),
                    lane_len=round(_wlane_len, 1),
                    next_route_edge=_wnext,
                    cur_lane_links=_wlinks[:5],
                )
            elif silent_done:
                # Other silent failure (true silent_stop / stuck) —
                # existing handling.
                cost += 1.0
                reward -= R_COLLISION_PENALTY + R_COLLISION_SPEED_BONUS * self._collision_speed_factor()
                done = True
                term_reason = silent_reason
        # Per-step shaping: progress + speed tracking + conditional idle + rule
        # violations. On terminal steps we only apply the rule-violation cost
        # (shaping signals are meaningless at termination).
        if not done:
            reward += self._reward_v2_shaped()
            cost += self._rule_violation_cost()

        # 2026-05-02 (round-7): accumulate per-episode stats for EPISODE-SUMMARY.
        try:
            cur_speed_for_avg = float(traci.vehicle.getSpeed(EGO_ID)) \
                if EGO_ID in traci.vehicle.getIDList() else 0.0
        except Exception:
            cur_speed_for_avg = 0.0
        self._ep_speed_sum += cur_speed_for_avg
        self._ep_speed_n += 1
        self._ep_reward_sum += float(reward)
        self._ep_cost_sum += float(cost)
        # 2026-05-03 (round-7 #4): merge-channel telemetry per-episode.
        if abs(lane_signal) > LANE_SIGNAL_THRESHOLD:
            self._ep_merge_cmds += 1
        # Detect actual lane changes via lane index transitions.
        try:
            cur_lane_idx_now = int(traci.vehicle.getLaneIndex(EGO_ID))
        except Exception:
            cur_lane_idx_now = self._ep_prev_lane_idx
        if (self._ep_prev_lane_idx >= 0 and cur_lane_idx_now >= 0
                and cur_lane_idx_now != self._ep_prev_lane_idx):
            self._ep_real_lane_changes += 1
        self._ep_prev_lane_idx = cur_lane_idx_now

        obs = self._observe(is_first=False, is_terminal=done, cost=float(cost))
        if _dbg().count("env_step") % 300 == 0 and _dbg().enabled:
            _dbg().kv(reward=reward, cost=cost, done=done)
            _dbg().stats("obs.ego", obs["ego"])
            _dbg().stats("obs.neighbors", obs["neighbors"])
            _dbg().stats("obs.route_ctx", obs["route_ctx"])
            _dbg().stats("obs.preferred_lane", obs["preferred_lane"])
        # 2026-05-03 (round-7 #4): periodic verification that our preferred_lane
        # obs matches SUMO's getBestLanes ground truth. Fires every 100 calls
        # via the env_pl_check counter (separate from env_step). Cheap cross-check.
        if _dbg().every("preferred_lane_check", 100) and EGO_ID in traci.vehicle.getIDList():
            try:
                bl = traci.vehicle.getBestLanes(EGO_ID)
                cur_lane = traci.vehicle.getLaneID(EGO_ID)
                cur_idx = traci.vehicle.getLaneIndex(EGO_ID)
                cur_edge = traci.vehicle.getRoadID(EGO_ID)
                n_lanes = traci.edge.getLaneNumber(cur_edge) if not cur_edge.startswith(":") else -1
                correct_indices_sumo = []
                for li in bl:
                    lid = li[0]
                    if not lid.startswith(":") and li[4]:  # allowsContinuation
                        try:
                            correct_indices_sumo.append(int(lid.rsplit("_", 1)[-1]))
                        except Exception:
                            pass
                _dbg().header(
                    "PREFERRED-LANE-CHECK",
                    edge=cur_edge,
                    n_lanes=n_lanes,
                    cur_lane=cur_lane,
                    cur_idx=cur_idx,
                    sumo_correct_lanes=sorted(correct_indices_sumo),
                    obs_is_correct=round(float(obs["preferred_lane"][0]), 2),
                    obs_offset_norm=round(float(obs["preferred_lane"][1]), 3),
                    obs_n_correct_frac=round(float(obs["preferred_lane"][2]), 3),
                )
            except Exception as e:
                _dbg().header("PREFERRED-LANE-CHECK", error=repr(e))
        # Always log the first step after a terminal so we can see *why* each
        # episode ended and confirm the reward bookkeeping is right.
        if done and term_reason is not None:
            _dbg().header(
                "TERMINAL",
                step=self._step_idx,
                reason=term_reason,
                reward=round(reward, 3),
                cost=round(cost, 3),
            )
            # 2026-05-02 (round-7): EPISODE-SUMMARY — one line per episode
            # so the operator can see at a glance: how long, how fast,
            # what happened, scenario phase. Single-line format makes it
            # easy to grep across long log files.
            avg_speed = self._ep_speed_sum / max(1, self._ep_speed_n)
            # 2026-05-03 (round-7 #4): merge channel ratio for the episode.
            # cmd_ratio = fraction of steps where the actor commanded a lane
            # change (|signal|>0.5). real_lc = number of lane-id transitions
            # (actual changes that happened). futile_rate = fraction of
            # commands that didn't cause a real change.
            cmd_ratio = self._ep_merge_cmds / max(1, self._step_idx)
            futile_rate = (
                (self._ep_merge_cmds - self._ep_real_lane_changes)
                / max(1, self._ep_merge_cmds)
            )
            _dbg().header(
                "EPISODE-SUMMARY",
                episode=self._episode_count,
                t_insert=round(self._ep_t_insert, 1),
                steps=self._step_idx,
                avg_speed=round(avg_speed, 2),
                total_reward=round(self._ep_reward_sum, 2),
                total_cost=round(self._ep_cost_sum, 2),
                reason=term_reason,
                success=(term_reason == "arrived" or term_reason == "route_end"),
                merge_cmds=self._ep_merge_cmds,
                real_lane_changes=self._ep_real_lane_changes,
                cmd_ratio=round(cmd_ratio, 3),
                futile_rate=round(futile_rate, 3),
            )
            # Collision-detail enhancement: when collision is the cause,
            # emit a separate line with full diagnostic.
            if term_reason in ("collision_at_fault", "collision_victim"):
                try:
                    coll_speed = float(traci.vehicle.getSpeed(EGO_ID)) \
                        if EGO_ID in traci.vehicle.getIDList() else float(self._last_speed)
                    coll_lane = traci.vehicle.getLaneID(EGO_ID) \
                        if EGO_ID in traci.vehicle.getIDList() else "?"
                    coll_edge = traci.vehicle.getRoadID(EGO_ID) \
                        if EGO_ID in traci.vehicle.getIDList() else "?"
                except Exception:
                    coll_speed, coll_lane, coll_edge = float("nan"), "?", "?"
                _dbg().header(
                    "COLLISION-DETAIL",
                    fault=term_reason,
                    speed_at_impact=round(coll_speed, 2),
                    lane=coll_lane,
                    edge=coll_edge,
                    sim_time=round(traci.simulation.getTime(), 1),
                )
        info = {"step_idx": self._step_idx, "term_reason": term_reason}
        return obs, float(reward), bool(done), info

    def close(self):
        self._close()

    # ----------------------- helpers -------------------------------------
    def _close(self):
        if self._running:
            try:
                traci.close()
            except Exception:
                pass
        self._running = False

    def _curriculum_max_route_length(self) -> float:
        """Return the route-length cap for the current curriculum stage.

        Stage selection is by `self._episode_count` (incremented in reset()).
        See CURRICULUM_STAGES at module top for the schedule.
        """
        for upper, max_len in CURRICULUM_STAGES:
            if upper is None or self._episode_count < upper:
                return float(max_len)
        return float("inf")  # safety fallback (shouldn't reach)

    def _select_ego_insertion_time(self) -> float:
        """Pick a SUMO sim time at which to schedule ego insertion.

        Sampled uniformly from [t_min, t_max] for the curriculum stage
        keyed by `self._episode_count`. Hard-capped at INSERT_TIME_MAX
        so we never schedule ego after the trip file has drained.

        2026-05-02 (round-7). Replaces _curriculum_warmup_seconds. Together
        with _queue_ego_at + _wait_for_insertion this realises the queue-
        based ego entry: ego is added to SUMO with depart=t_insert, then
        the env steps the simulation until ego is actually inserted.
        """
        for upper, t_min, t_max in EGO_INSERT_CURRICULUM:
            if upper is None or self._episode_count < upper:
                t = self._rng.uniform(float(t_min), float(t_max))
                return min(INSERT_TIME_MAX, t)
        # Should not reach; fall back to a safe small value.
        return 60.0

    def _compute_route_correct_lanes(self, from_edge, route_edges):
        """Pre-insertion analogue of getBestLanes: returns the set of lane
        indexes on `from_edge` whose outgoing connection reaches the route's
        next edge (route_edges[1]).

        Reasoning: SUMO's `departLane="best"` is NOT route-aware. It picks by
        speed/free-space heuristics and can place ego on a lane whose downstream
        link goes to an off-route edge. Audit (Phase 5) showed ~11% of failed
        episodes had outright-wrong spawns and 24% had tight (frac<1) spawns.
        Using `lane.getLinks()` here lets us filter to lanes that actually
        connect to the route's next edge BEFORE calling `vehicle.add`.

        2-edge lookahead: when the route has ≥3 edges, prefer lanes whose
        connection at edges[1] also has an onward connection to edges[2]
        (avoids tight scenarios where ego must lane-change immediately
        after the first junction).

        Returns:
            (correct_lane_idxs, n_lanes, deep_lane_idxs)
              - correct_lane_idxs: lanes on from_edge connecting to route_edges[1]
              - n_lanes: total lanes on from_edge
              - deep_lane_idxs: subset of correct_lane_idxs whose downstream
                lane on route_edges[1] also connects to route_edges[2] (or
                equal to correct_lane_idxs when route has only 2 edges)

        2026-05-05 (round-7 #9 F1).
        """
        try:
            n_lanes = traci.edge.getLaneNumber(from_edge)
        except Exception:
            return ([], 0, [])
        if n_lanes <= 0 or len(route_edges) < 2:
            return ([], n_lanes, [])
        next_edge = route_edges[1]
        deeper_edge = route_edges[2] if len(route_edges) >= 3 else None
        correct = []
        deep = []
        for i in range(n_lanes):
            lane_id = f"{from_edge}_{i}"
            try:
                links = traci.lane.getLinks(lane_id)
            except Exception:
                continue
            connects_to_next = False
            connects_two_deep = False
            for link in links:
                # link[0] is the destination lane id (e.g., "9940781_1"); the
                # link tuple may have additional fields (via, hasPrio, etc.).
                to_lane = link[0] if link else ""
                if not to_lane or to_lane.startswith(":"):
                    continue
                to_edge = to_lane.rsplit("_", 1)[0]
                if to_edge != next_edge:
                    continue
                connects_to_next = True
                if deeper_edge is None:
                    connects_two_deep = True
                else:
                    # Recurse one step: does to_lane connect to deeper_edge?
                    try:
                        for link2 in traci.lane.getLinks(to_lane):
                            tl2 = link2[0] if link2 else ""
                            if not tl2 or tl2.startswith(":"):
                                continue
                            if tl2.rsplit("_", 1)[0] == deeper_edge:
                                connects_two_deep = True
                                break
                    except Exception:
                        pass
            if connects_to_next:
                correct.append(i)
                if connects_two_deep:
                    deep.append(i)
        return (correct, n_lanes, deep)

    def _pick_route(self):
        """Select a random (from_edge, to_edge) pair feasible on the subregion.

        Lane-count rotation (added 2026-04-27, F-OBS-5): pure random sampling
        from the edge list under-exposed minority lane counts. The brief-train
        audit found 0 of 471 steps spent on 2-lane edges (6 of 38 drivable
        edges in the scenario) under seed=42. To guarantee balanced exposure
        across lane counts, we bias every spawn so that EITHER `from` or `to`
        belongs to the lane-count bucket selected by `self._spawn_idx %
        len(buckets)`. The route between the two ends is then up to SUMO's
        shortest-path; we don't enforce that the *path* contains every lane
        count, just the endpoints.

        Round-5 (2026-04-30): also applies a route-length curriculum filter
        keyed by self._episode_count. Early episodes get short routes (≤
        1500 m) so the agent can experience arrivals; the cap relaxes in
        stages and is removed after ~300 episodes.
        """
        # M8 (2026-05-02): exclude known dead-end edges from spawn / destination
        # candidates. The 6 edges in DEAD_END_EDGES generated 95% of all
        # ego emergency-stop events in the round-6 log audit (286 total).
        all_edges = [
            e for e in traci.edge.getIDList()
            if not e.startswith(":") and e not in DEAD_END_EDGES
        ]
        if len(all_edges) < 2:
            raise RuntimeError("Subregion has <2 drivable edges")

        # Bucket edges by lane count.
        buckets: dict[int, list[str]] = {}
        for e in all_edges:
            try:
                n = traci.edge.getLaneNumber(e)
            except Exception:
                continue
            buckets.setdefault(n, []).append(e)
        # Sorted lane-count list — only consider buckets with ≥1 edge.
        lc_list = sorted(buckets.keys())
        target_lc = lc_list[self._spawn_idx % max(len(lc_list), 1)]
        biased_pool = buckets.get(target_lc, [])
        self._spawn_idx += 1

        max_route_len = self._curriculum_max_route_length()

        for attempt in range(40):  # was 20 — extra headroom for curriculum filter
            if biased_pool and attempt < 24:
                # Force one endpoint to be in the target-lane bucket.
                a = self._rng.choice(biased_pool)
                b = self._rng.choice(all_edges)
                if a == b:
                    continue
            else:
                # Fallback: pure random sample from all edges.
                a, b = self._rng.sample(all_edges, 2)
            # Avoid recent (a, b) pairs: skip if seen in the last 200 spawns.
            # We allow exact reverse (b, a) as a different route — that's a
            # different driving experience. Only rejecting identical pairs.
            if (a, b) in self._recent_routes:
                continue
            try:
                route = traci.simulation.findRoute(a, b)
                if not (route and route.edges):
                    continue
                # M8 (2026-05-02): also skip routes whose path TRAVERSES a
                # known dead-end edge — not just routes whose endpoints land
                # on one. This catches the case where the routing planner
                # picks a path that goes through a problematic intermediate
                # edge.
                if any(e in DEAD_END_EDGES for e in route.edges):
                    continue
                # Curriculum filter: skip routes longer than the current
                # stage's cap. `route.length` is total route length in metres.
                rlen = float(getattr(route, "length", 0.0))
                if rlen <= 0.0:
                    # Older SUMO versions don't populate `length`; sum edge lens
                    # instead so the curriculum filter still works.
                    rlen = 0.0
                    for eid in route.edges:
                        try:
                            rlen += float(traci.lane.getLength(eid + "_0"))
                        except Exception:
                            pass
                if rlen > max_route_len:
                    continue
                self._recent_routes.append((a, b))
                # Round-5: per-route arrival bonus (variant d). Easy short
                # routes get smaller bonuses, hard long routes proportion-
                # ally bigger ones, but no route is unrewarded.
                self._arrival_bonus_for_episode = (
                    R_ARRIVAL_BONUS_BASE + R_ARRIVAL_BONUS_PER_M * rlen
                )
                return a, b, route.edges
            except Exception:
                continue
        raise RuntimeError("No feasible ego route found in 40 attempts")

    def _queue_ego_at(self, t_insert: float):
        """Schedule ego for insertion via SUMO's native vehicle queue.

        Adds ego with depart=t_insert (a future sim time) instead of
        depart="now". SUMO will defer insertion until sim_time >= t_insert
        AND a safe gap exists at the chosen lane. After this call, the
        caller must step the simulation (see _wait_for_insertion) until
        EGO_ID appears in traci.vehicle.getIDList().

        2026-05-02 (round-7). Replaces immediate-spawn path of the prior
        _spawn_ego_with_retries. Returns the route edge list (used for
        diagnostics) or raises if no feasible route exists.
        """
        from_edge, to_edge, edge_list = self._pick_route()
        # 2026-05-05 (round-7 #9 F2): route-aware spawn-lane selection.
        # Replaces departLane="best" with an explicit lane index chosen so
        # that ego spawns on a lane that connects to the route's next edge.
        # See _compute_route_correct_lanes for full motivation.
        correct_idxs, n_lanes, deep_idxs = self._compute_route_correct_lanes(
            from_edge, edge_list
        )
        # Pick from the deepest set (2-edge lookahead) when available, fall
        # back to the 1-deep correct set, fall back to "best" if both empty
        # (defensive — should not happen for routes findRoute returned).
        depart_lane_str = "best"
        chosen_idx = -1
        choice_pool = deep_idxs if deep_idxs else correct_idxs
        if choice_pool:
            # Prefer lowest-index lane (typical "main thru" lane in highways).
            chosen_idx = min(choice_pool)
            depart_lane_str = str(chosen_idx)
        elif correct_idxs is not None and not correct_idxs:
            _dbg().header(
                "SPAWN-FALLBACK",
                from_edge=from_edge,
                next_edge=edge_list[1] if len(edge_list) >= 2 else "?",
                n_lanes=n_lanes,
                note="no route-correct lane found; falling back to 'best'",
            )
        # 2026-05-05 (round-7 #9 F5): adapt depart speed to lane speed limit.
        # Hardcoded 5 m/s caused immediate rear-ends on highways where leader
        # is doing 25-30 m/s. Cap at 5 for slow lanes; scale up to half the
        # speed limit on faster lanes (still well under flow speed).
        try:
            lane_id_for_speed = f"{from_edge}_{chosen_idx}" if chosen_idx >= 0 else f"{from_edge}_0"
            speed_lim = float(traci.lane.getMaxSpeed(lane_id_for_speed))
            adaptive_speed = min(max(5.0, speed_lim * 0.4), max(5.0, speed_lim * 0.5))
            depart_speed_str = f"{adaptive_speed:.2f}"
        except Exception:
            depart_speed_str = EGO_DEPART_SPEED
        # Each retry uses a fresh route id to avoid name collision with a
        # prior failed attempt (SUMO may keep stale route definitions).
        route_id = f"ego_route_ep{self._episode_count}_t{int(t_insert*10)}"
        try:
            traci.route.add(route_id, list(edge_list))
        except Exception:
            pass  # already exists from prior attempt — fine to reuse
        # Telemetry: spawn-choice rationale.
        frac = (len(correct_idxs) / max(n_lanes, 1)) if correct_idxs else 0.0
        _dbg().header(
            "SPAWN-CHOICE",
            episode=self._episode_count,
            from_edge=from_edge,
            n_lanes=n_lanes,
            correct_lane_idxs=correct_idxs,
            deep_lane_idxs=deep_idxs,
            chosen_idx=chosen_idx,
            departLane=depart_lane_str,
            departSpeed=depart_speed_str,
            frac=round(frac, 3),
        )
        traci.vehicle.add(
            EGO_ID,
            route_id,
            typeID=EGO_TYPE,
            depart=str(t_insert),          # ← KEY: queue for future sim time
            departLane=depart_lane_str,    # F2: route-correct lane (was "best")
            departSpeed=depart_speed_str,  # F5: adaptive (was hardcoded "5")
        )
        # Cache the chosen lane and frac for the F3 retry-on-tight logic
        # in reset() to inspect after route is queued.
        self._last_spawn_choice = {
            "from_edge": from_edge,
            "chosen_idx": chosen_idx,
            "correct_idxs": correct_idxs,
            "n_lanes": n_lanes,
            "frac": frac,
        }
        return from_edge, to_edge, edge_list

    def _wait_for_insertion(self, t_insert: float) -> bool:
        """Step simulation forward until ego appears in vehicle.getIDList()
        OR the per-attempt timeout elapses.

        Returns True if ego was successfully inserted, False otherwise.
        On False, the caller is responsible for cleanup (remove ego) and
        retrying with a different route or hard-failing.
        """
        deadline = float(t_insert) + QUEUE_TIMEOUT_PER_ATTEMPT
        cur_t = traci.simulation.getTime()
        # Phase A: fast-forward to t_insert (no model in the loop, sim-only)
        while cur_t < t_insert:
            traci.simulationStep()
            cur_t = traci.simulation.getTime()
            if EGO_ID in traci.vehicle.getIDList():
                # SUMO inserted ego earlier than t_insert (rare but possible
                # if depart parsing rounds down). Treat as success.
                return True
        # Phase B: wait up to QUEUE_TIMEOUT_PER_ATTEMPT for SUMO to find a
        # safe gap at the chosen lane. Each step is just SUMO physics, fast.
        while cur_t < deadline:
            traci.simulationStep()
            cur_t = traci.simulation.getTime()
            if EGO_ID in traci.vehicle.getIDList():
                return True
        return False

    def _post_insertion_setup(self, t_insert: float):
        """One-shot setup after ego has been inserted: subscribe context,
        disable safety overrides (so the actor is responsible), apply GUI
        styling. Also emits a [DBG SPAWN] telemetry line.
        """
        # Subscribe to context around ego for fast neighbor queries.
        traci.vehicle.subscribeContext(
            EGO_ID,
            tc.CMD_GET_VEHICLE_VARIABLE,
            NEIGHBOR_RADIUS,
            [
                tc.VAR_POSITION,
                tc.VAR_SPEED,
                tc.VAR_ANGLE,
                tc.VAR_LENGTH,
                tc.VAR_VEHICLECLASS,
                tc.VAR_SIGNALS,
            ],
        )
        # 2026-05-05 (round-7 #10): arm the spawn-guard counter. For the
        # next SPAWN_GUARD_STEPS step() calls, env vehicles within
        # ±SPAWN_GUARD_RADIUS of ego on the same lane will be vaporized
        # to prevent immediate-spawn rear-ends and "spawn-into-trap"
        # collisions where ego accelerates into a slow leader.
        self._spawn_guard_steps_remaining = SPAWN_GUARD_STEPS
        # 2026-05-03 (round-7 #4): FULL ego autonomy — both speed and lane
        # safety overrides are disabled. Strategic LC is now also OFF so
        # SUMO doesn't covertly trajectory-enforce lane positioning during
        # WM training. Instead, the new preferred_lane obs + per-step
        # bonus + smart futile penalty teach the actor to position itself.
        # When ego reaches a dead-end lane (wrong lane for next turn), it
        # gets a small wrong_lane_end terminal penalty (-10) — much less
        # than a collision but enough to be informative.
        #
        # setLaneChangeMode bitmask:
        #   bit 0  = strategic        ← disabled
        #   bit 1  = cooperative      ← disabled
        #   bit 2  = speed-gain       ← disabled
        #   bit 3  = right-of-way     ← disabled
        # Value = 0 means NO automatic lane-change interventions.
        try:
            traci.vehicle.setSpeedMode(EGO_ID, 0)
            traci.vehicle.setLaneChangeMode(EGO_ID, 0)
            # Verify the modes were actually set (debug check on first
            # episodes only to avoid log spam).
            if self._episode_count <= 3:
                try:
                    actual_speed_mode = traci.vehicle.getSpeedMode(EGO_ID)
                    actual_lc_mode = traci.vehicle.getLaneChangeMode(EGO_ID)
                except Exception:
                    actual_speed_mode = actual_lc_mode = -1
                _dbg().header(
                    "ENFORCE-CHECK",
                    episode=self._episode_count,
                    requested_speed_mode=0,
                    actual_speed_mode=actual_speed_mode,
                    requested_lc_mode=0,
                    actual_lc_mode=actual_lc_mode,
                    note="both should be 0 for full ego autonomy",
                )
        except Exception as e:
            print(f"Warning: could not disable ego safety overrides: {e!r}")
        # GUI styling
        if self._gui:
            try:
                traci.vehicle.setColor(EGO_ID, (255, 255, 0, 255))
                traci.vehicle.setWidth(EGO_ID, 3.0)
                traci.vehicle.setLength(EGO_ID, 6.5)
                traci.vehicle.setHeight(EGO_ID, 2.2)
            except Exception as e:
                print(f"Warning: could not style ego in GUI: {e!r}")
            if self._gui_track_ego:
                try:
                    view_id = traci.gui.getIDList()[0]
                    traci.gui.trackVehicle(view_id, EGO_ID)
                    traci.gui.setZoom(view_id, 1800)
                    traci.gui.setSchema(view_id, "real world")
                except Exception:
                    pass
        # Spawn telemetry
        try:
            actual_speed = traci.vehicle.getSpeed(EGO_ID)
            actual_lane = traci.vehicle.getLaneID(EGO_ID)
            n_vehicles = len(traci.vehicle.getIDList())
            sim_time = traci.simulation.getTime()
        except Exception:
            actual_speed, actual_lane, n_vehicles, sim_time = float("nan"), "?", -1, -1
        # Initialize self._last_speed to ego's actual insertion speed.
        # Without this, the first env step would compute accel_est =
        # (5 m/s - 0 m/s)/0.1s = 50 m/s² (saturated to ±30 by the M5 clip),
        # giving the encoder a misleading "huge accel" signal on step 0.
        # Found in round-7 verification trace.
        try:
            self._last_speed = float(actual_speed) if not np.isnan(actual_speed) else 0.0
        except Exception:
            self._last_speed = 0.0
        _dbg().header(
            "SPAWN",
            episode=self._episode_count,
            t_insert=round(t_insert, 1),
            actual_t=round(sim_time, 1),
            n_vehicles=n_vehicles,
            lane_id=actual_lane,
            speed=round(float(actual_speed), 2),
        )

    @staticmethod
    def _safe_gap(fetch_fn):
        """Normalise getLeader/getFollower return values to a positive gap
        in metres, returning +inf if no partner is within the search radius.
        getLeader returns (id, gap) or None; getFollower returns (id, dist) where
        id="" when no follower is visible.
        """
        try:
            r = fetch_fn()
        except Exception:
            return float("inf")
        if r is None:
            return float("inf")
        try:
            other, dist = r
        except Exception:
            return float("inf")
        if not other:
            return float("inf")
        if dist < 0:  # SUMO signals "none" with a negative distance in some versions
            return float("inf")
        return float(dist)

    def _apply_action(self, accel_norm: float, lane_signal: float):
        """Apply (accel, lane_signal) to ego.

        Round-5 (2026-04-30): also writes self._last_action_meta so
        _reward_v2_shaped can apply lane-change-discipline penalties.
        Three failure modes are tracked:
          - `futile_cmd`: |lane_signal|>0.5 but no fire (cooldown / edge / junction)
          - `oscillation`: cmd direction is opposite to last successful merge
            within MERGE_OSCILLATION_WINDOW seconds
          - `fired`: a real lane change was issued

        Only the futile/osc cases get a small reward penalty. A successful
        merge is free (already shaped through progress + speed reward).
        """
        # Reset meta each call — defaults to "no command, nothing to penalize".
        # 2026-05-03 (round-7 #4): also store lane_signal_value so the smart
        # direction-aware futile penalty can read it back.
        self._last_action_meta = {
            "fired": False,
            "futile_cmd": False,
            "oscillation": False,
            "lane_signal_value": float(lane_signal),
        }
        if EGO_ID not in traci.vehicle.getIDList():
            return
        # Override SUMO's default car-following accel
        a = MAX_ACCEL * accel_norm if accel_norm >= 0 else MAX_DECEL * accel_norm
        try:
            traci.vehicle.setAcceleration(EGO_ID, float(a), duration=STEP_LENGTH)
        except AttributeError:
            # older SUMO versions: set target speed instead
            cur = traci.vehicle.getSpeed(EGO_ID)
            traci.vehicle.setSpeed(EGO_ID, max(0.0, cur + a * STEP_LENGTH))

        # If |lane_signal| is below threshold, no command was made → skip
        # discipline accounting. The threshold acts as a "neutral zone".
        if abs(lane_signal) < LANE_SIGNAL_THRESHOLD:
            return

        cmd_dir = 1 if lane_signal > 0 else -1   # +1 = left, -1 = right
        now = traci.simulation.getTime()

        # Oscillation: opposite direction within the window — flag now (penalty
        # applies whether or not the command also fires). Independent of
        # cooldown so an attempted reversal is detected even mid-cooldown.
        if (
            self._last_merge_dir != 0
            and cmd_dir == -self._last_merge_dir
            and (now - self._last_lane_change_t) < MERGE_OSCILLATION_WINDOW
        ):
            self._last_action_meta["oscillation"] = True

        # Cooldown: command issued during the merge cooldown window. Mark
        # futile and skip the rest of the lane-change logic.
        if now - self._last_lane_change_t < LANE_CHANGE_COOLDOWN:
            self._last_action_meta["futile_cmd"] = True
            return

        # Junction: lane changes inside a junction are physically meaningless.
        cur_idx = traci.vehicle.getLaneIndex(EGO_ID)
        edge_id = traci.vehicle.getRoadID(EGO_ID)
        if edge_id.startswith(":"):
            self._last_action_meta["futile_cmd"] = True
            return

        # Edge lane / single-lane edge: command goes nowhere. Mark futile.
        num_lanes = traci.edge.getLaneNumber(edge_id)
        target = cur_idx + cmd_dir
        target = max(0, min(num_lanes - 1, target))
        if target == cur_idx:
            self._last_action_meta["futile_cmd"] = True
            return

        # Successful fire — actually issue the lane change.
        traci.vehicle.changeLane(EGO_ID, target, duration=LANE_CHANGE_COOLDOWN)
        self._last_lane_change_t = now
        self._last_merge_dir = cmd_dir
        self._last_action_meta["fired"] = True

    def _reward_v2_shaped(self):
        """Reward v2 (CarDreamer-style weights, SUMO-adapted):
          + progress along the intended route (dominant signal)
          + tent-shaped speed tracking, **capped at vmax** so speeding is
            never rewarded (previously the +0.1·min(v/vmax, 1.2) form
            rewarded up to 120% of the limit — that's what v1's policy
            collapse exploited)
          − idle penalty only when ego is stopped *without* a justified
            reason. "Justified" includes: red/yellow TL within 15 m,
            leader within 8 m (close-follow), leader within `v·2 s`
            time-headway, or leader within 50 m and stopped (queue).
            See `_is_justified_stop` for the full predicate.
        The terminal arrival bonus and collision penalty are applied in
        step() itself — they are much larger and dominate this shaping.
        """
        if EGO_ID not in traci.vehicle.getIDList():
            return 0.0
        # Progress along route (replaces the v1 speed-×-time shaping)
        try:
            cur_dist = float(traci.vehicle.getDistance(EGO_ID))
        except Exception:
            cur_dist = self._prev_route_dist
        progress_m = max(0.0, cur_dist - self._prev_route_dist)
        self._prev_route_dist = cur_dist
        r = R_PROGRESS_SCALE * progress_m

        # Speed tracking: peak at v=vmax, tent-shaped, zero at 0 and at 2·vmax.
        v = float(traci.vehicle.getSpeed(EGO_ID))
        # Round-6 perf: cache lane_id once for this function (was queried 3 times
        # at lines 993, 1052, 1066). Cuts traci RPC count for _reward_v2_shaped
        # by ~50% with bit-identical output.
        try:
            cached_lane_id = traci.vehicle.getLaneID(EGO_ID)
        except Exception:
            cached_lane_id = ""
        try:
            vmax = max(
                float(traci.lane.getMaxSpeed(cached_lane_id)) if cached_lane_id else 13.9,
                1e-3,
            )
        except Exception:
            vmax = 13.9
        if v <= vmax:
            speed_tracking = v / vmax  # 0 at rest → 1 at limit
        else:
            speed_tracking = max(0.0, 2.0 - v / vmax)  # 1 at limit → 0 at 2·vmax
        r += R_SPEED_SCALE * speed_tracking

        # Round-5 (2026-04-30): absolute speed penalties.
        # The pre-round-5 design only penalised speeding via the cost channel
        # and only when context was unsafe (leader/TLS nearby). On empty roads
        # speeding had ZERO consequence, so the agent learned to floor it
        # and reach 50-80 m/s on suburban streets. The two penalties below
        # apply CONSTANTLY (not context-conditional):
        #   1) limit-overage: speed > 1.15 × lane vmax → linear penalty
        #      ramping from 0 at 1.15× to R_LIMIT_OVERAGE_PENALTY at 2.0×
        #   2) absolute-cap: speed > ABS_SPEED_CAP (=30 m/s) → linear penalty
        #      ramping from 0 at 30 m/s to R_ABS_SPEED_PENALTY at 60 m/s
        # We take the max of the two so the more-egregious dominates.
        # Magnitudes intentionally small (≤ 1/3 of progress reward) so they
        # shape behaviour without dominating, and so the cost-channel ψ
        # remains the dominant safety signal.
        limit_ratio = v / vmax
        limit_excess = max(0.0, limit_ratio - (1.0 + LIMIT_OVERAGE_THRESHOLD))
        # Normalise so penalty saturates at 2.0× limit (limit_excess = 0.85)
        limit_pen = min(1.0, limit_excess / 0.85)
        abs_excess = max(0.0, v - ABS_SPEED_CAP)
        # Normalise so penalty saturates at 60 m/s (abs_excess = 30)
        abs_pen = min(1.0, abs_excess / 30.0)
        speed_penalty = max(
            R_LIMIT_OVERAGE_PENALTY * limit_pen,
            R_ABS_SPEED_PENALTY * abs_pen,
        )
        r -= speed_penalty

        # Conditional idle penalty — only punish *true stopping* without a
        # reason. Threshold dropped 0.5 → 0.1 m/s on 2026-04-28: the smoke
        # trace caught a real edge case where ego creeping at 0.47 m/s
        # (1.7 km/h, plausible heavy-traffic crawl) tripped the idle penalty
        # even though it was making forward progress. Net signal was
        # -0.014/step (idle penalty dominates), giving the actor a perverse
        # incentive to stop entirely rather than creep. CarDreamer /
        # Highway-Env both use ~0.1 m/s as the "stopped" boundary.
        # Note: the silent_failure `stuck` detector intentionally keeps the
        # 0.5 threshold (different purpose — catches sustained no-progress
        # that warrants terminating the episode, not per-step shaping).
        if v < 0.1 and not self._is_justified_stop():
            r -= R_IDLE_PENALTY

        # Cross-track-error penalty (round-4, M2). Drifting laterally inside
        # the lane gets a smooth pull back to centre. Dead zone = w/4 either
        # side of centre (so minor wiggle is free); penalty ramps to
        # -R_CTE_SCALE when ego is at the lane edge. Inside a junction this
        # call may fail or return junk → silent except keeps reward stable.
        try:
            lat_pos = abs(float(traci.vehicle.getLateralLanePosition(EGO_ID)))
            lane_w = float(traci.lane.getWidth(cached_lane_id)) if cached_lane_id else 3.2
            half_w = max(lane_w * 0.5, 1e-3)
            deadzone = half_w * 0.5  # quarter-lane on either side of center
            offset_excess = max(0.0, lat_pos - deadzone)
            cte_norm = min(1.0, offset_excess / max(half_w - deadzone, 1e-3))
            r -= R_CTE_SCALE * cte_norm
        except Exception:
            pass

        # Heading-vs-route alignment penalty (round-4, M3). Uses |sin| of
        # the angle between ego heading and the current lane's forward
        # direction. Identical computation to route_ctx[4] in `_observe`,
        # duplicated here so reward doesn't depend on stale obs state.
        # Round-6 perf: reuse cached_lane_id from top of fn.
        try:
            lane_id = cached_lane_id
            if not lane_id:
                raise ValueError("no lane_id")
            lane_pos = float(traci.vehicle.getLanePosition(EGO_ID))
            lane_len = max(float(traci.lane.getLength(lane_id)), 1e-3)
            shape = traci.lane.getShape(lane_id)
            if len(shape) >= 2:
                seg_t = min(max(lane_pos / lane_len, 0.0), 1.0)
                seg_idx = min(int(seg_t * (len(shape) - 1)), len(shape) - 2)
                p0, p1 = shape[seg_idx], shape[seg_idx + 1]
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                norm = math.hypot(dx, dy) or 1e-3
                lx, ly = dx / norm, dy / norm
                heading_deg = float(traci.vehicle.getAngle(EGO_ID))
                theta = math.radians(90.0 - heading_deg)
                cos_th = math.cos(theta)
                sin_th = math.sin(theta)
                # |sin(ego_heading - lane_dir)| ∈ [0, 1]
                heading_sin_offset = abs(cos_th * ly - sin_th * lx)
                r -= R_HEADING_SCALE * heading_sin_offset
        except Exception:
            pass

        # 2026-05-03 (round-7 #4 → #5 → #6 → #7): UNIFIED lane-shaping block.
        # All three lane-related rewards (per-step bonus, futile-merge penalty,
        # oscillation penalty) are gated on `lane_choice_matters` — i.e., whether
        # at least one lane on the current edge would lead OFF-route. When every
        # lane is route-correct (num_correct_frac >= 1.0), lane positioning is
        # genuinely irrelevant on this edge — no false reward, no false penalty.
        #
        # The gate is justified by getBestLanes semantics: a lane gets
        # allowsContinuation=False iff it doesn't connect (via the upcoming
        # junction) to the route's next edge. So num_correct_frac < 1.0 fires
        # exactly when the upcoming junction discriminates between lanes.
        #
        # Single TraCI call — was previously called twice per step.
        try:
            is_correct, offset_norm, num_correct_frac = self._compute_preferred_lane_info()
        except Exception:
            is_correct, offset_norm, num_correct_frac = 1.0, 0.0, 1.0
        lane_choice_matters = (num_correct_frac < 1.0 - 1e-3)

        # SMART direction-aware futile penalty — only when lane choice matters.
        # offset_norm points to the BEST lane (longest continuation). Penalty:
        #   - commanded TOWARD best lane → 0 penalty (intent is useful even
        #     if cooldown/junction/edge blocks the actual change)
        #   - already on best lane (offset ≈ 0) + commanded change → 2× (spam)
        #   - commanded AWAY from best lane → 3× (actively wrong intent)
        # If lane_choice_matters is False, ALL lanes are equivalent so a
        # commanded change is harmless — skip the penalty entirely.
        meta = self._last_action_meta
        if meta.get("futile_cmd", False) and lane_choice_matters:
            try:
                ls_value = float(meta.get("lane_signal_value", 0.0))
                cmd_left  = ls_value > LANE_SIGNAL_THRESHOLD
                cmd_right = ls_value < -LANE_SIGNAL_THRESHOLD
                if cmd_left and offset_norm > 0:
                    pass  # toward best → no penalty
                elif cmd_right and offset_norm < 0:
                    pass  # toward best → no penalty
                elif (cmd_left or cmd_right) and abs(offset_norm) < 1e-3:
                    # already on best lane, but commanded change → spam
                    r -= R_FUTILE_MERGE_PENALTY * 2.0
                elif (cmd_left and offset_norm < 0) or (cmd_right and offset_norm > 0):
                    # commanded AWAY from best lane → actively wrong
                    r -= R_FUTILE_MERGE_PENALTY * 3.0
                else:
                    r -= R_FUTILE_MERGE_PENALTY  # default fallback
            except Exception:
                r -= R_FUTILE_MERGE_PENALTY  # safe fallback on any error

        # Oscillation penalty — only when lane choice matters. On all-correct
        # edges, rapid LEFT-then-RIGHT is just exploration with no positioning
        # consequence; no need to suppress it.
        if meta.get("oscillation", False) and lane_choice_matters:
            r -= R_MERGE_OSCILLATION_PENALTY

        # Per-step lane-positioning bonus — only when lane choice matters.
        # Without the gate, ego would earn +0.005/step indefinitely on wide
        # multi-lane highways where every lane continues — i.e., free reward
        # for breathing. The gate restricts the bonus to cases where staying
        # in a correct lane is a real choice the agent had to make.
        if is_correct > 0.5 and lane_choice_matters:
            r += R_LANE_CORRECT_BONUS

        return float(r)

    def _collision_speed_factor(self) -> float:
        """Returns min(1, v / vmax) at the moment of impact for use in the
        speed-scaled collision penalty (round-4, M1).

        Falls back to `self._last_speed` when ego has already left the
        sim (the unsafe_removal branch fires after `EGO_ID not in
        getIDList()`), so speed-scaling still produces meaningful values
        in all three terminal branches.
        """
        try:
            v = float(traci.vehicle.getSpeed(EGO_ID))
            vmax = float(traci.lane.getMaxSpeed(traci.vehicle.getLaneID(EGO_ID)))
        except Exception:
            v = float(self._last_speed)
            vmax = 13.9
        return min(1.0, v / max(vmax, 1e-3))

    def _is_justified_stop(self) -> bool:
        """Ego is entitled to be stopped if ANY of the following holds:

          (a) Red/yellow TLS within 15 m  — original.
          (b) Leader within 8 m  — original "close follow" case.
          (c) Leader within `v_ego · 2 s` time-headway window (added
              2026-04-28). Covers prudent following distances: at
              v=15 m/s the window is 30 m, matching real-world 2-second
              rule. Floor at 8 m so case (b) is preserved at low speeds.
          (d) Leader within 50 m AND leader.speed < 1.0 m/s (added
              2026-04-28). Covers heavy-traffic queues — even if ego is
              still at a safe distance, stopping for a stopped queue
              ahead is correct, not "stuck."

        Used by the idle-penalty branch of `_reward_v2_shaped` (sumo.py:691)
        AND by `_detect_silent_failure` (sumo.py:746). Both branches
        automatically benefit from the extended logic — there are no other
        callers, so adjusting this predicate is sufficient (no cascade).

        Background: pre-2026-04-28 the leader window was 8 m, which forced
        the agent to tailgate to receive the idle exemption (8 m ≈ 0.5 s
        headway at 15 m/s, well below the 2-second standard). Stopping
        behind a stopped queue at, say, 15 m gap was treated as gridlock,
        triggering both the idle penalty and the 10-s `stuck` termination.
        See the conversation analysis dated 2026-04-28.
        """
        # (a) TLS branch — unchanged
        try:
            tls = traci.vehicle.getNextTLS(EGO_ID)
            if tls and len(tls) > 0:
                _tls_id, _idx, dist, state = tls[0]
                if dist < 15.0 and state and state[0].lower() in ("r", "y"):
                    return True
        except Exception:
            pass

        # (b)–(d) Leader branch — speed-and-time-headway aware
        try:
            # Look up to 60 m ahead — sufficient for queue detection while
            # still bounded so far-away vehicles never count.
            leader = traci.vehicle.getLeader(EGO_ID, 60.0)
            if leader and leader[0]:
                leader_id, gap = leader[0], float(leader[1])
                try:
                    cur_speed = float(traci.vehicle.getSpeed(EGO_ID))
                except Exception:
                    cur_speed = 0.0
                try:
                    leader_speed = float(traci.vehicle.getSpeed(leader_id))
                except Exception:
                    leader_speed = 0.0

                # (b) Close-follow — preserved exactly
                if gap < 8.0:
                    return True
                # (c) Time-headway window — 2 s at current ego speed,
                #     floored at 8 m for the stationary case.
                time_headway_window = max(8.0, cur_speed * 2.0)
                if gap < time_headway_window:
                    return True
                # (d) Stopped queue ahead — leader within 50 m and at
                #     near-zero speed. Triggers regardless of current gap.
                if gap < 50.0 and leader_speed < 1.0:
                    return True
        except Exception:
            pass
        return False

    def _compute_preferred_lane_info(self) -> tuple[float, float, float]:
        """Returns (is_correct, offset_to_BEST_normalized, num_correct_frac).

        Uses traci.vehicle.getBestLanes() — SUMO's own internal computation
        of which lanes can continue along the planned route. We don't
        replicate any logic; we just read SUMO's authoritative answer.

        - is_correct (0.0 / 1.0): is current lane one of the route-correct lanes?
          (i.e., does it allow continuation along the planned route)
        - offset_normalized ∈ [-1, +1]: signed lane offset to the BEST lane
          (the one with the LONGEST continuation length, per getBestLanes).
          Positive = need to go LEFT, negative = need to go RIGHT, 0 = at the
          best lane already. Normalized by max-lanes-per-edge=5.
          NOTE (round-7 #5): even when is_correct=1, this offset can be non-zero
          (we're on a "valid" but not "optimal" lane). Encourages migration
          to the lane with longest continuation, reducing future required
          lane changes downstream.
        - num_correct_frac ∈ [0, 1]: fraction of current edge's lanes that allow
          continuation (1.0 = all-lanes-work / single-lane edge; small = tight
          constraint, multi-lane edge with few valid lanes).

        Returns safe defaults (1, 0, 1) if ego is not in the world or any
        traci call fails. Safe defaults mean "current lane is correct, on
        the best lane" so no incorrect penalty is ever applied due to
        query failures.

        2026-05-03 (round-7 #4 → #5: offset semantic refined to BEST).
        """
        if EGO_ID not in traci.vehicle.getIDList():
            return (1.0, 0.0, 1.0)
        try:
            cur_lane = traci.vehicle.getLaneID(EGO_ID)
            cur_idx = traci.vehicle.getLaneIndex(EGO_ID)
            best_lanes = traci.vehicle.getBestLanes(EGO_ID)
            if not best_lanes:
                return (1.0, 0.0, 1.0)
            # First pass: gather correct lane indices and their continuation
            # lengths. Tuple format from getBestLanes:
            #   (laneID, length, occupation, offset, allowsContinuation, bestLaneIDs)
            correct_lengths = {}  # lane_idx → length, for lanes with allowsContinuation
            for lane_info in best_lanes:
                lane_id = lane_info[0]
                length = lane_info[1]
                allows_cont = lane_info[4]
                if not lane_id or lane_id.startswith(":"):
                    continue
                try:
                    lane_idx_here = int(lane_id.rsplit("_", 1)[-1])
                except Exception:
                    continue
                if allows_cont:
                    correct_lengths[lane_idx_here] = float(length)
            if not correct_lengths:
                # Degenerate: no continuation possible from any lane on this edge.
                return (0.0, 0.0, 0.0)
            correct_indices = sorted(correct_lengths.keys())
            is_correct = (cur_idx in correct_lengths)
            # Round-7 #5: offset to BEST lane (longest continuation).
            # Round-7 #6: TIE-BREAKING — when multiple correct lanes tie at the
            # max length (e.g., all lanes equally good on a straight multi-lane
            # edge), PREFER the current lane. Otherwise we'd falsely tell the
            # agent to migrate to a different equally-good lane.
            max_length = max(correct_lengths.values())
            if is_correct and correct_lengths.get(cur_idx, -1.0) >= max_length - 1e-3:
                # Current lane is tied for max length → it IS the best, offset=0.
                best_idx = cur_idx
            else:
                # Current lane is either wrong, or correct-but-suboptimal.
                # Pick the index closest to cur_idx among lanes tied at max_length
                # (so the agent moves the SHORTEST distance to a best lane).
                tied_best = [i for i, L in correct_lengths.items()
                             if L >= max_length - 1e-3]
                best_idx = min(tied_best, key=lambda i: abs(i - cur_idx))
            offset_signed = best_idx - cur_idx  # positive = LEFT, negative = RIGHT
            offset_norm = max(-1.0, min(1.0, offset_signed / 5.0))
            try:
                cur_edge = traci.vehicle.getRoadID(EGO_ID)
                n_lanes = traci.edge.getLaneNumber(cur_edge)
            except Exception:
                n_lanes = max(len(best_lanes), 1)
            num_correct_frac = len(correct_indices) / max(n_lanes, 1)
            return (1.0 if is_correct else 0.0, float(offset_norm), float(num_correct_frac))
        except Exception:
            return (1.0, 0.0, 1.0)

    def _detect_wrong_lane_end(self) -> bool:
        """Returns True iff ego is at (within 5m of) the end of a lane that
        has NO connection to the next route edge.

        Used to distinguish 'wrong-lane-end' terminations (a navigation
        failure: agent should have positioned in a correct lane earlier)
        from 'silent_stop' (a safety/control failure). The two get
        different penalty magnitudes.

        Returns False (= it's NOT wrong_lane_end) on any failure of the
        underlying traci queries — falls back to the existing silent_stop
        path so no event is misclassified into the wrong bucket.

        2026-05-03 (round-7 #4).
        """
        try:
            if EGO_ID not in traci.vehicle.getIDList():
                return False
            lane_id = traci.vehicle.getLaneID(EGO_ID)
            if not lane_id or lane_id.startswith(":"):
                return False  # junction-internal lane
            lane_pos = float(traci.vehicle.getLanePosition(EGO_ID))
            lane_len = float(traci.lane.getLength(lane_id))
            if lane_pos < lane_len - 5.0:
                return False  # not near end
            route = traci.vehicle.getRoute(EGO_ID)
            route_idx = traci.vehicle.getRouteIndex(EGO_ID)
            if route_idx + 1 >= len(route):
                return False  # last edge of route — no "next" to fail on
            next_edge = route[route_idx + 1]
            # Does current lane have a connection to next_edge?
            for link in traci.lane.getLinks(lane_id):
                next_lane_id = link[0]
                if not next_lane_id:
                    continue
                next_edge_id = next_lane_id.rsplit("_", 1)[0]
                if next_edge_id == next_edge:
                    return False  # has a connection — not wrong_lane_end
            # No link to next_edge — this IS wrong_lane_end
            return True
        except Exception:
            return False

    def _detect_silent_failure(self) -> tuple[bool, str | None]:
        """Catch failure modes that the env's primary collision / removal
        branches miss when the ego is technically still 'alive' in SUMO:

        1) **silent_stop** — speed dropped from ≥ SUDDEN_STOP_PRIOR_SPEED
           to ≤ SUDDEN_STOP_CUR_SPEED in one step with no justification
           (no red/yellow TL nearby, no leader nearby). Physically impossible
           given MAX_DECEL=5 m/s² × dt=0.1 s = 0.5 m/s/step. So a drop > 5 m/s
           in one step IS a SUMO event we can't otherwise see (junction
           internal-edge clamp, soft jam, off-road).
        2) **stuck** — the ego has been stationary (speed < 0.5 m/s) without
           justification for STUCK_THRESHOLD_STEPS consecutive steps. Catches
           gridlocks and dead-end stops that wouldn't otherwise terminate.

        Returns (done: bool, term_reason: str | None). The caller is
        responsible for adding cost + reward penalty so these failures feed
        the cost head and the safety machinery.

        Note: only called from step() AFTER the ego_alive / ego_collided
        branches; safe to assume ego is in vehicle.getIDList() here.
        """
        if EGO_ID not in traci.vehicle.getIDList():
            return False, None  # already handled upstream

        try:
            cur_speed = float(traci.vehicle.getSpeed(EGO_ID))
        except Exception:
            return False, None

        justified = self._is_justified_stop()

        # (1) Sudden silent stop
        if (self._last_speed > SUDDEN_STOP_PRIOR_SPEED
                and cur_speed <= SUDDEN_STOP_CUR_SPEED
                and not justified):
            _dbg().header(
                "SILENT-FAILURE",
                kind="silent_stop",
                prev_speed=round(self._last_speed, 2),
                cur_speed=round(cur_speed, 2),
            )
            return True, "silent_stop"

        # (2) Prolonged unjustified idle
        if cur_speed < 0.5 and not justified:
            self._stuck_steps += 1
            if self._stuck_steps >= STUCK_THRESHOLD_STEPS:
                _dbg().header(
                    "SILENT-FAILURE",
                    kind="stuck",
                    stuck_steps=self._stuck_steps,
                    cur_speed=round(cur_speed, 2),
                )
                return True, "stuck"
        else:
            # Any non-trivial motion (or a justified stop) resets the counter.
            self._stuck_steps = 0

        return False, None

    def _rule_violation_cost(self):
        """Per-step cost for rule violations.

        Context-aware speeding cost (added 2026-04-28): the previous flat
        rule "v > 1.1·vmax → cost+=1" couldn't teach the cost critic to
        distinguish "empty road" from "crowded approach to intersection,"
        so SafeDreamer's *state-conditional* λ_p·V_c(s) penalty couldn't
        learn the context. Speeding is now classed by severity:

          - mild speeding (1.10 < v/vmax ≤ 1.30) on a CLEARLY-SAFE state
            (no leader within 50 m, no upcoming red TLS within 50 m, far
            from the next junction) costs nothing — lets the cost critic
            learn that going fast on empty road is fine.
          - mild speeding in any unsafe state (leader nearby, intersection
            approach, red TLS) costs +1 (preserves the original rule).
          - severe speeding (v > 1.3·vmax) always costs +1, regardless of
            context (an absolute hard limit — never safe).

        Implements the user's design intent: "speed on empty roads, slow
        down when traffic / intersections are present."
        """
        if EGO_ID not in traci.vehicle.getIDList():
            return 0.0
        cost = 0.0

        # Compute context flags once — reused below
        is_clearly_safe = self._is_speeding_context_safe()

        # Speeding logic (replaces the flat 1.1·vmax rule)
        try:
            v = traci.vehicle.getSpeed(EGO_ID)
            lane_id = traci.vehicle.getLaneID(EGO_ID)
            vmax = traci.lane.getMaxSpeed(lane_id)
            ratio = v / max(vmax, 1e-3)
            if ratio > 1.30:
                cost += 1.0   # severe — always a violation
            elif ratio > 1.10 and not is_clearly_safe:
                cost += 1.0   # mild speeding only counts when context is risky
        except Exception:
            pass

        # Red-light run: next TLS state is 'r' or 'y' and ego is within 3 m
        try:
            tls_list = traci.vehicle.getNextTLS(EGO_ID)
            if tls_list:
                _tls_id, _idx, dist, state = tls_list[0]
                if dist < 3.0 and state in ("r", "y", "R", "Y"):
                    cost += 1.0
        except Exception:
            pass
        return float(cost)

    def _spawn_guard_clear_neighbors(self):
        """Spawn-guard: for SPAWN_GUARD_STEPS step() calls after ego enters,
        remove any env vehicle on ego's CURRENT lane within ±SPAWN_GUARD_RADIUS.

        Complements `_remove_close_newcomers` (which only catches vehicles
        that just departed/spawned) — this one catches vehicles that
        ARRIVE (drive into) the danger zone via normal traffic flow.

        Two failure modes this prevents:
          (1) Fast follower already on ego's lane catches up and rear-ends
              the slow-spawned ego  →  collision_victim (-20, +0.3 cost).
          (2) Slow leader already in front of ego: when ego accelerates
              from EGO_DEPART_SPEED, it can rear-end them in 1-2 sim steps
              before the actor's policy has time to react
              →  collision_at_fault (-50, +1.0 cost) — actively damaging.

        Both are SPAWN-INDUCED (caused by ego entering at a low speed
        relative to the surrounding traffic) — not the agent's fault.
        Removing the threatening vehicle for ~2s gives ego time to
        accelerate to traffic speed.

        After SPAWN_GUARD_STEPS expire, the agent must handle traffic
        on its own. Counter is set in `_post_insertion_setup` and
        decremented every step() call.

        2026-05-05 (round-7 #10).
        """
        if self._spawn_guard_steps_remaining <= 0:
            return
        self._spawn_guard_steps_remaining -= 1
        if EGO_ID not in traci.vehicle.getIDList():
            return
        try:
            ego_lane = traci.vehicle.getLaneID(EGO_ID)
            if not ego_lane or ego_lane.startswith(":"):
                return  # junction-internal lane — skip (lane-pos comparisons ill-defined)
            ego_pos = float(traci.vehicle.getLanePosition(EGO_ID))
        except Exception:
            return
        removed = []
        try:
            for vid in traci.lane.getLastStepVehicleIDs(ego_lane):
                if vid == EGO_ID:
                    continue
                try:
                    their_pos = float(traci.vehicle.getLanePosition(vid))
                    gap = their_pos - ego_pos  # >0 ahead, <0 behind
                    if abs(gap) <= SPAWN_GUARD_RADIUS:
                        traci.vehicle.remove(vid, reason=tc.REMOVE_VAPORIZED)
                        removed.append((vid, round(gap, 1)))
                except Exception:
                    pass
        except Exception:
            return
        if removed:
            _dbg().header(
                "SPAWN-GUARD",
                episode=self._episode_count,
                step=self._step_idx,
                ego_lane=ego_lane,
                ego_pos=round(ego_pos, 1),
                removed_count=len(removed),
                removed=removed[:5],
                steps_remaining=self._spawn_guard_steps_remaining,
            )

    def _remove_close_newcomers(self):
        """Phase 3 isolation: remove any env vehicle that just inserted
        too close to the ego on the same lane.

        Called immediately after each `traci.simulationStep()` while ego
        is alive. Uses `getDepartedIDList()` to scope to vehicles that
        materialised THIS step (cheap; the list is short — typical
        density 0.5–1.5 inserts per step). Any newcomer whose lane
        position is within `EGO_NO_SPAWN_BUFFER` (= 30 m) of ego on the
        same lane is vaporised via `traci.vehicle.remove(...,
        REMOVE_VAPORIZED)`. The traffic city loses ≤ 0.2 % of its
        scheduled vehicles per training run — negligible compared to
        the 5896 trips/h volume.

        Logs every removal at DBG so audit logs can verify the mechanism
        fires only when ego is actually nearby.
        """
        if EGO_ID not in traci.vehicle.getIDList():
            return
        try:
            ego_lane = traci.vehicle.getLaneID(EGO_ID)
            ego_pos = float(traci.vehicle.getLanePosition(EGO_ID))
        except Exception:
            return
        try:
            just_departed = traci.simulation.getDepartedIDList()
        except Exception:
            return
        for vid in just_departed:
            if vid == EGO_ID:
                continue
            try:
                if traci.vehicle.getLaneID(vid) != ego_lane:
                    continue
                other_pos = float(traci.vehicle.getLanePosition(vid))
            except Exception:
                continue
            if abs(other_pos - ego_pos) < EGO_NO_SPAWN_BUFFER:
                try:
                    traci.vehicle.remove(vid, reason=tc.REMOVE_VAPORIZED)
                    _dbg().header(
                        "EGO-ISOLATION",
                        removed=vid,
                        ego_lane=ego_lane,
                        ego_pos=round(ego_pos, 1),
                        other_pos=round(other_pos, 1),
                        gap=round(abs(other_pos - ego_pos), 1),
                    )
                except Exception:
                    pass

    def _is_speeding_context_safe(self) -> bool:
        """Returns True if the current state has no nearby risk that would
        make speeding unsafe. Mirrors the structure of `_is_justified_stop`
        (proactive context check) but applied to acceleration rather than
        idling.

        Conditions ALL must hold for "clearly safe":
          - No same-lane leader within 50 m (gap longer than ~3 s @ 15 m/s)
          - No upcoming TLS within 50 m (red OR yellow OR green — any TLS
            ahead means an intersection where speed must drop)
          - Distance to next junction ≥ 50 m (so we are not approaching a
            merge / crossing where lane connections matter)
        """
        try:
            leader = traci.vehicle.getLeader(EGO_ID, 60.0)
            if leader and leader[0]:
                if float(leader[1]) < 50.0:
                    return False
        except Exception:
            return False
        try:
            tls_list = traci.vehicle.getNextTLS(EGO_ID)
            if tls_list:
                _, _, tls_dist, _state = tls_list[0]
                if float(tls_dist) < 50.0:
                    return False
        except Exception:
            return False
        try:
            lane_id = traci.vehicle.getLaneID(EGO_ID)
            lane_pos = traci.vehicle.getLanePosition(EGO_ID)
            lane_len = traci.lane.getLength(lane_id)
            if (lane_len - lane_pos) < 50.0:
                return False
        except Exception:
            return False
        return True

    def _observe(self, is_first: bool, is_terminal: bool, cost: float):
        nb_dim = N_NEIGHBORS * NEIGHBOR_FEAT_DIM
        tl_dim = N_TLS_AHEAD * TL_FEAT_DIM
        # When ego has left the sim (collision/teleport/arrival), return a
        # zero-padded obs but preserve is_first/is_terminal/cost.
        if EGO_ID not in traci.vehicle.getIDList():
            return {
                "ego": np.zeros(EGO_FEAT_DIM, np.float32),
                "neighbors": np.zeros(nb_dim, np.float32),
                "lights": np.zeros(tl_dim, np.float32),
                "route_ctx": np.zeros(ROUTE_CTX_DIM, np.float32),
                # 2026-05-03 (round-7 #4): preferred_lane defaults to (1, 0, 1)
                # when ego is not in world — interpreted as "current lane is fine,
                # no offset needed, all-correct" so reward shaping doesn't trigger.
                "preferred_lane": np.array([1.0, 0.0, 1.0], dtype=np.float32),
                "is_first": np.bool_(is_first),
                "is_terminal": np.bool_(is_terminal),
                "cost": np.float32(cost),
            }

        # --- ego features ---
        speed = traci.vehicle.getSpeed(EGO_ID)
        accel_est = (speed - self._last_speed) / STEP_LENGTH
        # Round-6 (2026-05-02): clip accel_est to ±3 × MAX_OBS_ACCEL.
        # SUMO returns speed=0 instantly on collision impact, so a 27 m/s
        # cruise → 0 transition produces accel_est = -270 m/s² (= -27 normalized,
        # observed in [DBG WM-BATCH] data.ego min). Clipping at ±30 m/s² (= ±3
        # normalized) preserves real hard-braking (~5 m/s²) and crash-impact
        # signal (~10-20 m/s²) while killing the unphysical numerical-artifact
        # tail that destabilizes encoder normalization. _last_speed is still
        # updated to raw speed so sudden-stop detection (line ~1225) is
        # unaffected.
        accel_est = max(-3.0 * MAX_OBS_ACCEL, min(3.0 * MAX_OBS_ACCEL, accel_est))
        self._last_speed = speed
        lane_id = traci.vehicle.getLaneID(EGO_ID)
        lane_idx = traci.vehicle.getLaneIndex(EGO_ID)
        lane_pos = traci.vehicle.getLanePosition(EGO_ID)
        edge_id = traci.vehicle.getRoadID(EGO_ID)
        try:
            lane_len = traci.lane.getLength(lane_id)
            vmax = traci.lane.getMaxSpeed(lane_id)
            lane_width = traci.lane.getWidth(lane_id)
        except Exception:
            lane_len, vmax, lane_width = 1.0, 13.9, 3.2
        if edge_id.startswith(":"):
            # Inside a junction — lane-count concept doesn't apply, use 1.
            # Clamp lane_idx to 0 too (F-OBS-2, 2026-04-27): SUMO returns the
            # internal lane index (0, 1, 2, ...) which would otherwise
            # produce a logically inconsistent obs like "lane 1 of 1 lanes."
            num_lanes = 1
            lane_idx = 0
        else:
            try:
                num_lanes = traci.edge.getLaneNumber(edge_id)
            except Exception:
                num_lanes = 1
        lane_pos_norm = lane_pos / max(lane_len, 1e-3)

        heading_deg = traci.vehicle.getAngle(EGO_ID)
        # Math frame conversion: SUMO compass (0=N, +clockwise) → math (0=E, +CCW).
        theta = math.radians(90.0 - heading_deg)
        cos_th, sin_th = math.cos(theta), math.sin(theta)
        heading_sin = math.sin(math.radians(heading_deg))
        heading_cos = math.cos(math.radians(heading_deg))

        # Coarse road-type categorical (added 2026-04-27).
        # Mapping (in [0, 1]):
        #   0.00 → internal/junction edge   (edge_id starts with ':')
        #   0.33 → residential / low-speed  (vmax ≤ 14 m/s ≈ 50 km/h)
        #   0.66 → secondary / tertiary     (14 < vmax ≤ 22 m/s ≈ 80 km/h)
        #   1.00 → primary / highway        (vmax > 22 m/s)
        # This is a coarse hint on top of the raw vmax (already in ego[7]); it
        # gives the encoder a categorical signal that's faster to learn than
        # inferring road class from continuous speed-limit thresholds.
        if edge_id.startswith(":"):
            road_type = 0.0
        elif vmax <= 14.0:
            road_type = 1.0 / 3.0
        elif vmax <= 22.0:
            road_type = 2.0 / 3.0
        else:
            road_type = 1.0

        # Ego length (F-OBS-3, 2026-04-27). Constant per episode for the
        # default ego type but exposed so the policy can reason about its own
        # rear-bumper position when computing distances to followers.
        try:
            ego_length = float(traci.vehicle.getLength(EGO_ID))
        except Exception:
            ego_length = 5.0
        # Round-5 (2026-04-30): pre-compute "I'm at an edge lane" booleans.
        # SUMO indexes lanes from 0 (rightmost in US right-hand traffic) to
        # num_lanes-1 (leftmost). Inside a junction num_lanes is clamped to
        # 1 → both flags are 1.0 there, but the merge handler rejects all
        # lane changes inside junctions anyway, so this is harmless.
        is_at_right_edge = 1.0 if lane_idx == 0 else 0.0
        is_at_left_edge = 1.0 if lane_idx == num_lanes - 1 else 0.0
        # All ego features normalized so the encoder sees inputs in roughly
        # [-1, 1]. heading_{sin,cos} and lane_pos_norm are already bounded;
        # everything else is divided by a configured maximum (see top-of-file).
        ego = np.array(
            [
                speed / MAX_OBS_SPEED,
                accel_est / MAX_OBS_ACCEL,
                heading_sin,
                heading_cos,
                float(lane_idx) / MAX_OBS_LANE,
                float(num_lanes) / MAX_OBS_LANE,
                lane_pos_norm,
                vmax / MAX_OBS_SPEED,
                lane_width / MAX_OBS_LEN,
                road_type,                          # index 9
                ego_length / MAX_OBS_LEN,           # index 10
                is_at_left_edge,                    # NEW index 11 (round-5)
                is_at_right_edge,                   # NEW index 12 (round-5)
            ],
            dtype=np.float32,
        )

        # --- neighbors in Frenet-in-ego-frame ---
        sub = traci.vehicle.getContextSubscriptionResults(EGO_ID) or {}
        ego_pos = traci.vehicle.getPosition(EGO_ID)
        # Ego velocity in math frame (for relative-velocity decomposition)
        ego_vx, ego_vy = speed * cos_th, speed * sin_th
        nbs = []
        for vid, attrs in sub.items():
            if vid == EGO_ID:
                continue
            pos = attrs.get(tc.VAR_POSITION, (0.0, 0.0))
            sp = attrs.get(tc.VAR_SPEED, 0.0)
            ang = attrs.get(tc.VAR_ANGLE, 0.0)
            length = float(attrs.get(tc.VAR_LENGTH, 5.0))
            vclass = attrs.get(tc.VAR_VEHICLECLASS, "passenger")
            class_id = {"passenger": 0.0, "bus": 1.0, "motorcycle": 2.0}.get(
                vclass, -1.0
            )
            # Turn-signal bitmask (F-OBS-4): bit 0 = right blinker, bit 1 = left.
            # Collapse to a signed scalar: +1 left, -1 right, 0 none/both.
            sig_bits = int(attrs.get(tc.VAR_SIGNALS, 0))
            blink_right = bool(sig_bits & 0x1)
            blink_left = bool(sig_bits & 0x2)
            if blink_left and not blink_right:
                signal_feat = 1.0
            elif blink_right and not blink_left:
                signal_feat = -1.0
            else:
                signal_feat = 0.0
            # Neighbor lane index — for the lane_delta feature. Note that
            # vehicle.getLaneIndex isn't part of the subscribed context vars
            # (we'd have to subscribe VAR_LANE_INDEX), so do a direct call.
            # Inside an internal/junction edge we cannot meaningfully compare
            # lane indices to ego's, so fall back to ego's index (delta=0).
            try:
                nb_lane_idx = traci.vehicle.getLaneIndex(vid)
                if not edge_id.startswith(":"):
                    try:
                        nb_edge = traci.vehicle.getRoadID(vid)
                        if nb_edge.startswith(":") or nb_edge != edge_id:
                            # Different edge — lane-index comparison isn't
                            # well-defined; conservatively report delta=0.
                            nb_lane_idx = lane_idx
                    except Exception:
                        nb_lane_idx = lane_idx
            except Exception:
                nb_lane_idx = lane_idx
            # Lane delta — neighbor's lane index minus ego's, divided by
            # MAX_OBS_LANE (=5) and clipped to [-1, 1] (F-OBS-6, 2026-04-27).
            # Previously divided by 3, which saturated for any lane gap of 4+
            # (matters on real arterials). Same denominator as ego[4]/[5].
            lane_delta = max(
                -1.0, min(1.0, (nb_lane_idx - lane_idx) / MAX_OBS_LANE)
            )
            dx = pos[0] - ego_pos[0]
            dy = pos[1] - ego_pos[1]
            # Rotate (dx, dy) into the ego heading frame: long_off forward,
            # lat_off to ego's left. This is the key lane-invariant step.
            long_off = dx * cos_th + dy * sin_th
            lat_off = -dx * sin_th + dy * cos_th
            # Neighbor velocity in math frame
            nb_theta = math.radians(90.0 - ang)
            nb_vx = sp * math.cos(nb_theta)
            nb_vy = sp * math.sin(nb_theta)
            dvx, dvy = nb_vx - ego_vx, nb_vy - ego_vy
            rel_v_long = dvx * cos_th + dvy * sin_th
            rel_v_lat = -dvx * sin_th + dvy * cos_th
            heading_diff_sin = math.sin(math.radians(ang - heading_deg))
            # Manhattan-Frenet rank: penalises lateral separation less than
            # Euclidean, so same-lane leaders/followers are always ranked high.
            rank_key = abs(long_off) + abs(lat_off)
            # Normalized: long/lat offsets by NEIGHBOR_RADIUS (~[-1, 1]),
            # rel velocities by MAX_REL_V (=2·MAX_OBS_SPEED) so head-on
            # closure rates stay in [-1, 1] (F-OBS-1, 2026-04-27),
            # length by MAX_OBS_LEN, class_id by 2.0 → {-0.5, 0.0, 0.5, 1.0}
            # for {pad, pass, bus, moto}.
            # heading_diff_sin and presence are already bounded.
            # lane_delta is computed above — already in [-1, 1].
            # signal_feat ∈ {-1, 0, +1}.
            nbs.append(
                (
                    rank_key,
                    [
                        1.0,  # presence
                        long_off / NEIGHBOR_RADIUS,
                        lat_off / NEIGHBOR_RADIUS,
                        rel_v_long / MAX_REL_V,
                        rel_v_lat / MAX_REL_V,
                        heading_diff_sin,
                        length / MAX_OBS_LEN,
                        class_id / 2.0,
                        lane_delta,                  # index 8
                        signal_feat,                 # NEW (index 9)
                    ],
                )
            )
        nbs.sort(key=lambda t: t[0])
        feats = [f for _, f in nbs[:N_NEIGHBORS]]
        while len(feats) < N_NEIGHBORS:
            # Padded slot: presence=0, class_id sentinel=-0.5 (= -1/2 normalized
            # class space), lane_delta=0 (neutral), signal=0 (none).
            # Length = NEIGHBOR_FEAT_DIM = 10.
            feats.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0])
        neighbors = np.array(feats, dtype=np.float32).reshape(-1)

        # --- upcoming traffic lights ---
        try:
            tls_list = traci.vehicle.getNextTLS(EGO_ID)[:N_TLS_AHEAD]
        except Exception:
            tls_list = []
        tl_feats = []
        for _tls_id, _idx, dist, state in tls_list:
            ch = (state or "r")[0].lower()
            # Distance clipped at MAX_OBS_DIST then normalized to [0, 1].
            # Beyond MAX_OBS_DIST the agent saturates to 1.0; not a problem
            # because near-TLS reactions matter, far ones don't.
            norm_dist = min(float(dist), MAX_OBS_DIST) / MAX_OBS_DIST
            tl_feats.extend(
                [
                    norm_dist,
                    1.0 if ch == "r" else 0.0,
                    1.0 if ch == "y" else 0.0,
                    1.0 if ch == "g" else 0.0,
                    0.0,  # time-to-switch placeholder; requires TL id lookup
                ]
            )
        while len(tl_feats) < tl_dim:
            tl_feats.append(0.0)
        lights = np.array(tl_feats, dtype=np.float32)

        # --- route context (merge / goal / heading vs route) ---
        route_ctx = self._compute_route_ctx(
            edge_id=edge_id,
            lane_id=lane_id,
            lane_pos=lane_pos,
            lane_len=lane_len,
            num_lanes=num_lanes,
            lane_idx=lane_idx,
            cos_th=cos_th,
            sin_th=sin_th,
        )

        # 2026-05-03 (round-7 #4): compute preferred_lane info from SUMO's
        # getBestLanes — bit-identical to SUMO's strategic-LC computation,
        # which we have now disabled (setLaneChangeMode=0).
        is_correct, offset_norm, n_correct_frac = self._compute_preferred_lane_info()
        preferred_lane = np.array(
            [is_correct, offset_norm, n_correct_frac], dtype=np.float32
        )

        return {
            "ego": ego,
            "neighbors": neighbors,
            "lights": lights,
            "route_ctx": route_ctx,
            "preferred_lane": preferred_lane,
            "is_first": np.bool_(is_first),
            "is_terminal": np.bool_(is_terminal),
            "cost": np.float32(cost),
        }

    # ---------------- route context helpers ------------------------------
    HORIZON_M = 300.0  # look ahead up to 300 m along the route

    def _edge_lane_count(self, edge):
        if edge.startswith(":"):
            return None
        try:
            return traci.edge.getLaneNumber(edge)
        except Exception:
            return None

    def _edge_default_length(self, edge):
        if edge.startswith(":"):
            return 0.0
        try:
            return traci.lane.getLength(f"{edge}_0")
        except Exception:
            return 0.0

    def _compute_route_ctx(
        self,
        edge_id: str,
        lane_id: str,
        lane_pos: float,
        lane_len: float,
        num_lanes: int,
        lane_idx: int,
        cos_th: float,
        sin_th: float,
    ):
        rc = np.zeros(ROUTE_CTX_DIM, dtype=np.float32)

        # NOTE: rc[0], rc[1], rc[3] are distances; we keep the raw accumulators
        # in metres internally (so the loop logic against HORIZON_M still works)
        # and normalize ONLY the value written into rc. This keeps the
        # accumulation precise and the obs in [-1, 1].

        # [0] distance to end of current lane (≈ distance to next junction).
        # Stored in metres for the loop below; normalized form goes into rc[0].
        dist_to_next_junction = float(max(0.0, lane_len - lane_pos))
        rc[0] = min(dist_to_next_junction, MAX_OBS_DIST) / MAX_OBS_DIST

        # [1,2] distance to next lane-count change + its delta (merge/split).
        # The -1 sentinel stays as -1 in rc[1] (no clip / no normalize) so the
        # encoder can distinguish "no change in horizon" from "0 m to change".
        dist_to_change = -1.0
        lane_count_delta = 0.0
        try:
            route = traci.vehicle.getRoute(EGO_ID)
            route_idx = traci.vehicle.getRouteIndex(EGO_ID)
            accum = dist_to_next_junction          # metres
            for i in range(route_idx + 1, len(route)):
                nxt = route[i]
                nxt_count = self._edge_lane_count(nxt)
                if nxt_count is None:
                    continue
                if nxt_count != num_lanes:
                    dist_to_change = float(accum)   # metres
                    lane_count_delta = float(nxt_count - num_lanes)
                    break
                accum += self._edge_default_length(nxt)
                if accum > self.HORIZON_M:
                    break
        except Exception:
            pass
        rc[1] = (
            -1.0
            if dist_to_change < 0
            else min(dist_to_change, MAX_OBS_DIST) / MAX_OBS_DIST
        )
        # Normalize lane_count_delta to [-1, 1] for consistency with the rest
        # of the observation. Typical values are ±1, ±2 (lane count rarely
        # changes by more than 3 between connected edges); divide by 3 and
        # clip. Without this, rc[2] could reach ±2.0 which violated the
        # encoder's [-1, 1] expectation (caught in smoke §6 sample inspection).
        rc[2] = max(-1.0, min(1.0, lane_count_delta / 3.0))

        # [3] distance remaining on route — clipped + normalized to [0, 1].
        try:
            route = traci.vehicle.getRoute(EGO_ID)
            route_idx = traci.vehicle.getRouteIndex(EGO_ID)
            remaining = dist_to_next_junction       # metres
            for i in range(route_idx + 1, len(route)):
                remaining += self._edge_default_length(route[i])
            rc[3] = min(remaining, MAX_OBS_DIST) / MAX_OBS_DIST
        except Exception:
            rc[3] = 0.0

        # [4] sine of the angle between ego heading and the current lane's
        # forward direction (zero when aligned). Approximates "heading error
        # relative to the route."
        try:
            shape = traci.lane.getShape(lane_id)
            if len(shape) >= 2 and lane_len > 1e-3:
                # Pick the segment the ego is currently on.
                seg_t = min(max(lane_pos / lane_len, 0.0), 1.0)
                seg_idx = min(int(seg_t * (len(shape) - 1)), len(shape) - 2)
                p0, p1 = shape[seg_idx], shape[seg_idx + 1]
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                norm = math.hypot(dx, dy) or 1e-3
                lx, ly = dx / norm, dy / norm
                # sin of angle = cross product z-component between ego forward
                # and lane forward.
                rc[4] = float(cos_th * ly - sin_th * lx)
        except Exception:
            rc[4] = 0.0

        # [5] target lane delta — how many lane indices the ego should shift
        # to stay on the route. Heuristic: if next edge has fewer lanes, shift
        # toward the inside; if it has more, 0; otherwise 0. Proper solution
        # requires inspecting connections; this is a cheap approximation that
        # still gives the agent a merge-direction signal.
        try:
            route = traci.vehicle.getRoute(EGO_ID)
            route_idx = traci.vehicle.getRouteIndex(EGO_ID)
            if route_idx + 1 < len(route):
                nxt_count = self._edge_lane_count(route[route_idx + 1])
                if nxt_count is not None and nxt_count < num_lanes:
                    # Shift toward the rightmost (lane 0) that survives.
                    rc[5] = float(-min(lane_idx, num_lanes - nxt_count))
        except Exception:
            rc[5] = 0.0

        return rc
