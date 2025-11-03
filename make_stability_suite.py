# make_stability_suite.py
# Robust stability/disturbance suite for PPO, DQN, and Hybrid-V2 (adaptive arbiter).
# - Safe episode metrics (no NaN explosions in summaries)
# - No use of .reset() on SB3 models
# - Disturbances specified in SECONDS (auto-converted to steps via env.dt)
# - Capacity integrated robustly if env/info don't provide it
# - Clean CSVs + plots

import os, json, math, csv, argparse
from collections import deque
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3 import DQN as SB3_DQN

# --------- Import env ----------
try:
    from gym_bms import BatteryPackEnv
except Exception as e:
    raise SystemExit(f"[FATAL] Could not import gym_bms.BatteryPackEnv: {e}")

# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def soc_variance_from_obs(obs: Any, cells: int = 5) -> float:
    """Estimate var(SoC) from the first N features if observation contains cell SoCs there."""
    try:
        arr = np.array(obs, dtype=float).reshape(-1)
        return float(np.var(arr[:cells]))
    except Exception:
        return float("nan")

def get_env_dt(env) -> float:
    return float(getattr(env, "dt", 1.0))

def get_pack_current(env, info) -> float:
    """Absolute pack current (A), robust to sign conventions and info/env variants."""
    if isinstance(info, dict) and ("I_load" in info):
        return abs(safe_float(info["I_load"]))
    if hasattr(env, "I_load"):
        return abs(safe_float(getattr(env, "I_load")))
    if hasattr(env, "_profile_current"):
        try:
            return abs(float(env._profile_current(env.t)))
        except Exception:
            pass
    return 0.0

def count_switch_changes(prev_bits, bits) -> int:
    if prev_bits is None or bits is None:
        return 0
    try:
        a = np.array(prev_bits, dtype=int).reshape(-1)
        b = np.array(bits, dtype=int).reshape(-1)
        return int(np.sum(a != b))
    except Exception:
        return 0

# =========================
# Hybrid-V2 (adaptive gate)
# =========================
class HybridV2:
    """
    Softmax/logistic-gated arbiter between PPO and DQN.
    - DQN probability increases with SoC variance above theta.
    - Min-span on each expert to avoid chatter.
    - Optional budget for how often we allow expert switches.
    """
    def __init__(self, ppo: SB3_PPO, dqn: SB3_DQN,
                 k: float = 70.0, theta: float = 0.0040,
                 window: int = 12, decay: float = 0.98,
                 max_switch_budget: int = 45,
                 min_dqn_span: int = 10, min_ppo_span: int = 6):
        self.ppo = ppo
        self.dqn = dqn
        self.k = float(k)
        self.theta = float(theta)
        self.window = int(window)
        self.decay = float(decay)
        self.max_switch_budget = int(max_switch_budget)
        self.min_dqn_span = int(min_dqn_span)
        self.min_ppo_span = int(min_ppo_span)

        self._hist = deque(maxlen=self.window)
        self._expert = "ppo"
        self._span_left = self.min_ppo_span
        self._budget_left = self.max_switch_budget

    def reset(self):
        self._hist.clear()
        self._expert = "ppo"
        self._span_left = self.min_ppo_span
        self._budget_left = self.max_switch_budget

    def _p_dqn(self, var_now: float) -> float:
        # Logistic on (var_now - theta)
        z = self.k * (var_now - self.theta)
        try:
            p = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            p = 1.0 if z > 0 else 0.0
        return float(p)

    def predict(self, obs, var_est: float):
        # Update variance history (unused mean kept for possible extensions)
        self._hist.append(var_est)

        # Budget decay when stable
        if var_est < self.theta:
            self._budget_left = max(0, self._budget_left - 0)  # keep for future; decay disabled here

        # Respect min span for current expert
        if self._span_left > 0:
            self._span_left -= 1
            expert = self._expert
        else:
            # Decide whether to switch expert
            if self._budget_left <= 0:
                expert = self._expert  # budget exhausted: keep current
            else:
                p_dqn = self._p_dqn(var_est)
                chosen = "dqn" if p_dqn >= 0.5 else "ppo"
                if chosen != self._expert:
                    self._expert = chosen
                    self._budget_left -= 1
                    self._span_left = self.min_dqn_span if chosen == "dqn" else self.min_ppo_span
                expert = self._expert

        if expert == "dqn":
            act, _ = self.dqn.predict(obs, deterministic=True)
            return int(act), "dqn"
        else:
            act, _ = self.ppo.predict(obs, deterministic=True)
            return int(act), "ppo"

# =========================
# Disturbance model (seconds → steps)
# =========================
class Disturber:
    """
    Disturbances are specified in SECONDS and converted to STEPS.
    Supported hooks are best-effort: the env must expose attributes below to take effect.
      - env.disturb_current_delta (A)
      - env.sensor_noise_sigma (SoC noise)
      - env.force_cell_off = (cell_index:int, is_active:bool)
    If these are absent, the disturbance call is a no-op (safe).
    """
    def __init__(self, events_sec: List[Dict], dt: float):
        self.dt = float(max(dt, 1e-9))
        self.events = []
        for ev in events_sec:
            t0s = float(ev.get("t_s", 0.0))
            durs = float(ev.get("dur_s", 0.0))
            self.events.append({
                "kind": ev["kind"],
                "t": int(round(t0s / self.dt)),
                "dur": max(1, int(round(durs / self.dt))),
                "mag": ev.get("mag", 0.0),
                "cell": ev.get("cell", 0)
            })

    def apply(self, env, step_i: int):
        for ev in self.events:
            t0 = ev["t"]; dur = ev["dur"]
            active = (step_i >= t0) and (step_i < t0 + dur)
            kind = ev["kind"]

            if kind in ("step_current", "pulse_current", "ramp_current"):
                if hasattr(env, "disturb_current_delta"):
                    if kind == "ramp_current":
                        if active:
                            frac = (step_i - t0) / max(1, dur)
                            env.disturb_current_delta = float(ev["mag"]) * float(frac)
                        else:
                            env.disturb_current_delta = 0.0
                    else:
                        env.disturb_current_delta = float(ev["mag"]) if active else 0.0

            elif kind == "noise_soc":
                if hasattr(env, "sensor_noise_sigma"):
                    env.sensor_noise_sigma = float(ev["mag"]) if active else 0.0

            elif kind == "open_cell":
                if hasattr(env, "force_cell_off"):
                    env.force_cell_off = (int(ev["cell"]), bool(active))

    def any_active(self, step_i: int) -> bool:
        for ev in self.events:
            if (step_i >= ev["t"]) and (step_i < ev["t"] + ev["dur"]):
                return True
        return False

# =========================
# Episode runner
# =========================
def run_episode(env: BatteryPackEnv,
                controller: Any,
                max_steps: int,
                disturbance_events_sec: List[Dict],
                recov_thresh: float = 0.0040,
                recov_window_steps: int = 15) -> Dict:
    """
    Returns robust per-episode metrics (floats or NaN):
      capacity_mAh, peak_variance, recovery_time_s, switches_total
    """
    reset_out = env.reset(seed=None)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if hasattr(controller, "reset"):
        controller.reset()

    dt = get_env_dt(env)
    dist = Disturber(disturbance_events_sec, dt)
    done = False

    Q_Ah = 0.0
    peak_after = 0.0
    post_start = None
    recov_time_s = float("nan")

    last_bits = None
    switches_total = 0

    var_hist: List[float] = []
    time_hist: List[float] = []

    for step_i in range(max_steps):
        var_now = soc_variance_from_obs(obs)
        var_hist.append(var_now)
        time_hist.append(step_i * dt)

        # Apply disturbance (if env doesn't support hooks, it's a no-op)
        dist.apply(env, step_i)

        # Act
        if isinstance(controller, HybridV2):
            act, _ = controller.predict(obs, var_now)
        else:
            act, _ = controller.predict(obs, deterministic=True)

        # Pre-step switch count
        before = int(getattr(env, "switch_count", 0))
        obs, reward, terminated, truncated, info = env.step(act)
        after = int(getattr(env, "switch_count", 0))

        switches_total += (after - before)
        # If env exposes switch bits, add bit flips too
        if hasattr(env, "switch_on"):
            switches_total += count_switch_changes(last_bits, env.switch_on)
            last_bits = np.array(env.switch_on).copy()

        # Capacity integrate robustly if needed
        I = get_pack_current(env, info)
        Q_Ah += I * (dt / 3600.0)

        # Peak + recovery tracking (after first activation)
        is_active = dist.any_active(step_i)
        if is_active and (post_start is None):
            post_start = step_i
        if post_start is not None and post_start >= 0:
            if var_now > peak_after:
                peak_after = var_now
            # Recovery: last recov_window_steps all below threshold
            if len(var_hist) >= recov_window_steps:
                window_vals = var_hist[-recov_window_steps:]
                if all((v <= recov_thresh) for v in window_vals):
                    recov_time_s = (step_i - post_start) * dt
                    post_start = -1  # stop tracking

        done = bool(terminated or truncated)
        if done:
            break

    # Get final capacity (prefer env/info if available)
    capacity_mAh = None
    if isinstance(info, dict) and ("usable_capacity_mAh" in info):
        capacity_mAh = safe_float(info["usable_capacity_mAh"])
    elif hasattr(env, "usable_capacity_mAh"):
        capacity_mAh = safe_float(getattr(env, "usable_capacity_mAh"))
    if capacity_mAh is None or not np.isfinite(capacity_mAh):
        capacity_mAh = safe_float(Q_Ah * 1000.0)

    return {
        "capacity_mAh": safe_float(capacity_mAh),
        "peak_variance": safe_float(peak_after),
        "recovery_time_s": safe_float(recov_time_s),
        "switches_total": safe_float(switches_total),
        "traj": {
            "t": time_hist,
            "var": var_hist,
            "sw": [switches_total] * len(time_hist)  # cumulative (display only)
        }
    }

# =========================
# Summaries (NaN-safe)
# =========================
def summarize_episode_list(episodes: List[Dict]) -> Dict:
    if not episodes:
        return {
            "n": 0,
            "capacity_mean_mAh": np.nan, "capacity_std_mAh": np.nan,
            "peak_variance_mean": np.nan, "recovery_time_median_s": np.nan,
            "switches_mean": np.nan
        }

    def col(key):
        vals = []
        for e in episodes:
            v = e.get(key, np.nan)
            vals.append(np.nan if v is None else v)
        return np.array(vals, dtype=float)

    cap = col("capacity_mAh")
    peak = col("peak_variance")
    trec = col("recovery_time_s")
    sw = col("switches_total")

    out = {
        "n": len(episodes),
        "capacity_mean_mAh": float(np.nanmean(cap)) if np.isfinite(cap).any() else np.nan,
        "capacity_std_mAh": float(np.nanstd(cap))  if np.isfinite(cap).any() else np.nan,
        "peak_variance_mean": float(np.nanmean(peak)) if np.isfinite(peak).any() else np.nan,
        "recovery_time_median_s": float(np.nanmedian(trec)) if np.isfinite(trec).any() else np.nan,
        "switches_mean": float(np.nanmean(sw)) if np.isfinite(sw).any() else np.nan
    }
    return out

# =========================
# Plot helpers
# =========================
def plot_episode(outdir: str, tag: str, ep_idx: int, res: Dict):
    ensure_dir(outdir)
    t = np.array(res["traj"]["t"], dtype=float)
    v = np.array(res["traj"]["var"], dtype=float)
    sw = np.array(res["traj"]["sw"], dtype=float)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, v, label="SoC variance")
    ax[0].axhline(0.0040, ls="--", color="tab:gray", alpha=0.6, label="recovery threshold")
    ax[0].set_ylabel("Var(SoC)")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, sw, label="cumulative switches")
    ax[1].set_ylabel("Switches")
    ax[1].set_xlabel("Time (s)")
    fig.suptitle(f"{tag} — episode {ep_idx+1} — cap={res['capacity_mAh']:.1f}mAh, "
                 f"peak={res['peak_variance']:.5f}, t_recov={res['recovery_time_s'] if np.isfinite(res['recovery_time_s']) else float('nan'):.0f}s")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}_ep{ep_idx+1}.png"), dpi=140)
    plt.close(fig)

def plot_metric_bars(outdir: str, summary_rows: List[List[Any]], metric_idx: int, title: str, ylabel: str, filename: str):
    import pandas as pd
    ensure_dir(outdir)
    df = pd.DataFrame(summary_rows, columns=[
        "scenario","algo","capacity_mean_mAh","capacity_std_mAh","cap_delta_mean_mAh",
        "peak_variance_mean","recovery_time_median_s","switches_mean"
    ])
    piv = df.pivot(index="scenario", columns="algo", values=df.columns[metric_idx])
    order = [c for c in ["PPO", "DQN", "HybridA", "HybridB"] if c in piv.columns]
    piv = piv[order]
    ax = piv.plot(kind="bar", figsize=(10,4))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(outdir, filename), dpi=140)
    plt.close(ax.figure)

# =========================
# Main suite
# =========================
def run_suite(args):
    outroot = "results_stability"
    plots_dir = os.path.join(outroot, "plots")
    ensure_dir(outroot); ensure_dir(plots_dir)

    # Disturbances defined in SECONDS (converted later using env.dt)
    scenarios = [
        {"name":"step_up",      "events_sec":[{"kind":"step_current",  "t_s":900,  "dur_s":600, "mag":+2.0}]},
        {"name":"step_down",    "events_sec":[{"kind":"step_current",  "t_s":900,  "dur_s":600, "mag":-2.0}]},
        {"name":"pulse_train",  "events_sec":[{"kind":"pulse_current", "t_s":600,  "dur_s":300, "mag":+3.0},
                                              {"kind":"pulse_current", "t_s":1500, "dur_s":300, "mag":-3.0}]},
        {"name":"ramp_up",      "events_sec":[{"kind":"ramp_current",  "t_s":600,  "dur_s":1200,"mag":+3.0}]},
        {"name":"noise",        "events_sec":[{"kind":"noise_soc",     "t_s":900,  "dur_s":600, "mag":0.004}]},
        {"name":"open_cell_c2", "events_sec":[{"kind":"open_cell",     "t_s":900,  "dur_s":300, "cell":2}]},
    ]

    # Load models
    ppo = SB3_PPO.load(args.ppo_ckpt, custom_objects={
        "lr_schedule": (lambda *_, **__: 0.0), "clip_range": (lambda *_, **__: 0.2)
    })
    dqn = SB3_DQN.load(args.dqn_ckpt, custom_objects={
        "lr_schedule": (lambda *_, **__: 0.0), "exploration_schedule": (lambda *_, **__: 0.0)
    })
    hybridA = HybridV2(ppo, dqn, k=70.0, theta=0.0040, window=12, max_switch_budget=45, min_dqn_span=10, min_ppo_span=6)
    hybridB = HybridV2(ppo, dqn, k=28.0, theta=0.0049, window=20, max_switch_budget=25, min_dqn_span=6,  min_ppo_span=10)

    algos = [("PPO", ppo), ("DQN", dqn), ("HybridA", hybridA), ("HybridB", hybridB)]

    # Build a base env to compute dt and check horizons
    env0 = BatteryPackEnv(profile=args.profile)
    dt = get_env_dt(env0)
    horizon_s = args.max_steps * dt

    # Horizon checks: warn if any disturbance is beyond episode end
    for sc in scenarios:
        for ev in sc["events_sec"]:
            t_s = float(ev.get("t_s", 0.0))
            if t_s >= horizon_s:
                print(f"[WARN] Scenario '{sc['name']}': disturbance at t={t_s:.1f}s exceeds horizon {horizon_s:.1f}s. "
                      f"Increase --max-steps or reduce t_s.")

    # Baseline (no-disturbance) capacity per algo
    base_caps: Dict[str, float] = {}
    for name, ctrl in algos:
        env = BatteryPackEnv(profile=args.profile)
        res = run_episode(env, ctrl, args.max_steps, disturbance_events_sec=[])
        base_caps[name] = res["capacity_mAh"]

    # Run each scenario × algo × episodes
    events_rows = []  # per-episode
    summary_rows = [] # per-scenario

    for sc in scenarios:
        # collect per-scenario episodes by algo
        per_algo_eps: Dict[str, List[Dict]] = {"PPO": [], "DQN": [], "HybridA": [], "HybridB": []}
        for name, ctrl in algos:
            for ep in range(args.episodes):
                env = BatteryPackEnv(profile=args.profile)
                res = run_episode(env, ctrl, args.max_steps, disturbance_events_sec=sc["events_sec"])

                # add capacity delta vs baseline for the same algo
                base_cap = base_caps.get(name, np.nan)
                cap_delta = res["capacity_mAh"] - base_cap if (np.isfinite(res["capacity_mAh"]) and np.isfinite(base_cap)) else np.nan

                per_algo_eps[name].append({
                    "capacity_mAh": res["capacity_mAh"],
                    "peak_variance": res["peak_variance"],
                    "recovery_time_s": res["recovery_time_s"],
                    "switches_total": res["switches_total"],
                    "cap_delta_mAh": cap_delta
                })

                # Save episode plot
                tag = f"{sc['name']}_{name}"
                plot_episode(os.path.join("results_stability", "plots"), tag, ep, res)

                # Row for events CSV
                events_rows.append([
                    sc["name"], name, ep+1,
                    res["capacity_mAh"], base_cap, cap_delta,
                    res["peak_variance"], res["recovery_time_s"], res["switches_total"]
                ])

        # Summaries for this scenario
        for name in ["PPO", "DQN", "HybridA", "HybridB"]:
            eps = per_algo_eps[name]
            s = summarize_episode_list(eps)
            summary_rows.append([
                sc["name"], name,
                s["capacity_mean_mAh"], s["capacity_std_mAh"],
                float(np.nanmean([e["cap_delta_mAh"] for e in eps if np.isfinite(e["cap_delta_mAh"])])) if eps else np.nan,
                s["peak_variance_mean"], s["recovery_time_median_s"], s["switches_mean"]
            ])

    # Write CSVs
    ensure_dir("results_stability")
    with open(os.path.join("results_stability", "stability_events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario","algo","episode","capacity_mAh","baseline_capacity_mAh",
                    "capacity_delta_mAh","peak_variance","recovery_time_s","switches_total"])
        w.writerows(events_rows)

    with open(os.path.join("results_stability", "stability_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario","algo","capacity_mean_mAh","capacity_std_mAh","cap_delta_mean_mAh",
                    "peak_variance_mean","recovery_time_median_s","switches_mean"])
        w.writerows(summary_rows)

    # Quick comparative bar charts
    try:
        # Recovery time
        plot_metric_bars(
            outdir=os.path.join("results_stability", "plots"),
            summary_rows=summary_rows,
            metric_idx=6,  # recovery_time_median_s
            title="Stability — Recovery Time (median) by Scenario",
            ylabel="Recovery Time (s)",
            filename="stability_recovery_time.png"
        )
        # Peak variance
        plot_metric_bars(
            outdir=os.path.join("results_stability", "plots"),
            summary_rows=summary_rows,
            metric_idx=5,  # peak_variance_mean
            title="Stability — Peak SoC Variance by Scenario",
            ylabel="Peak Variance",
            filename="stability_peak_variance.png"
        )
        # Switches
        plot_metric_bars(
            outdir=os.path.join("results_stability", "plots"),
            summary_rows=summary_rows,
            metric_idx=7,  # switches_mean
            title="Stability — Mean Switches by Scenario",
            ylabel="Mean Switches",
            filename="stability_switches.png"
        )
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    print("\n[Stability Suite] Done.")
    print(" - results_stability/stability_events.csv")
    print(" - results_stability/stability_summary.csv")
    print(" - results_stability/plots/*.png\n")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", required=True, help="Path to PPO .zip")
    ap.add_argument("--dqn-ckpt", required=True, help="Path to DQN .zip")
    ap.add_argument("--episodes", type=int, default=5, help="episodes per scenario")
    ap.add_argument("--max-steps", type=int, default=2400, help="max environment steps per episode")
    ap.add_argument("--profile", type=str, default="charge-rest-discharge")
    args = ap.parse_args()
    run_suite(args)
