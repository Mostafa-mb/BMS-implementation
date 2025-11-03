import os, argparse, csv, json, math
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from gym_bms import BatteryPackEnv
from stable_baselines3 import PPO, DQN


# ========== Failure Wrapper (no env source edits needed) ==========

class FailureWrapper:
    """
    Wraps a BatteryPackEnv to simulate an open-circuit failure of one cell
    without editing the environment sources.

    Effects from fail_time onward:
      - Forces the failed cell's switch OFF (bit = 0).
      - Freezes the failed cell's SoC at the fail instant (flat line).
      - Redistributes the SoC change that would have applied to the failed
        cell evenly to the remaining active cells (approximate current sharing).
    """
    def __init__(self, env: BatteryPackEnv, fail_cell: int, fail_time: float):
        self.env = env
        self.fail_cell = int(fail_cell)
        self.fail_time = float(fail_time)
        self.failed = False
        self._soc_freeze = None

    # convenience proxies
    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        self.failed = False
        self._soc_freeze = None
        return out

    def _force_bit(self, action: int, bit_index: int, value: int) -> int:
        mask = 1 << bit_index
        action &= ~mask
        if value:
            action |= mask
        return action

    def step(self, action: int):
        # trigger failure
        if (not self.failed) and (self.env.t >= self.fail_time):
            self.failed = True
            # capture SoC at failure to freeze that cell visually/physically
            self._soc_freeze = self.env.soc.copy()

        # if failed, force that cell switch OFF in the action
        if self.failed:
            action = self._force_bit(action, self.fail_cell, 0)

        # cache pre-step SoC, then step underlying env
        soc_before = self.env.soc.copy()
        obs, reward, done, trunc, info = self.env.step(action)
        soc_after = self.env.soc.copy()

        if self.failed:
            # how much SoC would have changed on the failed cell this step?
            d_failed = soc_after[self.fail_cell] - soc_before[self.fail_cell]

            # 1) freeze failed cell SoC at fail instant
            soc_after[self.fail_cell] = self._soc_freeze[self.fail_cell]

            # 2) mask switch of failed cell to off for clarity
            try:
                self.env.switch_on[self.fail_cell] = 0
            except Exception:
                pass

            # 3) redistribute the "lost" SoC change across remaining cells
            active_mask = np.ones_like(soc_after, dtype=bool)
            active_mask[self.fail_cell] = False
            n_active = int(active_mask.sum())
            if n_active > 0 and d_failed != 0.0:
                soc_after[active_mask] += d_failed / n_active

            # commit SoC back to env
            self.env.soc = soc_after

            # if env exposes a method to rebuild observation, use it
            if hasattr(self.env, "_get_obs"):
                obs = self.env._get_obs()

            # mark failure in info for downstream logging
            info = dict(info or {})
            info["failed_cell"] = self.fail_cell
            info["fail_time"] = self.fail_time

        return obs, reward, done, trunc, info

    # expose attributes of wrapped env transparently
    def __getattr__(self, name):
        return getattr(self.env, name)


# ========== Simple policies (NoBalancing / RuleBased) ==========

def act_nobalancing(env: BatteryPackEnv) -> int:
    return 0

def act_rule_based(env: BatteryPackEnv, I_load: float) -> int:
    tol = 0.01  # 1% SoC band
    bits = np.zeros(env.n, dtype=np.int32)
    mean_soc = float(np.mean(env.soc))
    if I_load > 0:  # discharge: bleed higher SoC
        for i in range(env.n):
            if env.soc[i] > mean_soc + tol:
                bits[i] = 1
    elif I_load < 0:  # charge: bleed lower SoC
        for i in range(env.n):
            if env.soc[i] < mean_soc - tol:
                bits[i] = 1
    action = 0
    for i, b in enumerate(bits):
        action |= (int(b) << i)
    return int(action)

def act_model(model, obs) -> int:
    a, _ = model.predict(obs, deterministic=True)
    return int(a)

def ms(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


# ========== Episode roll-out with wrapper ==========

def run_episode(policy: str, model, profile: str, seed: int,
                fail_cell: int, fail_time: float):
    base_env = BatteryPackEnv(seed=seed, profile=profile)
    env = FailureWrapper(base_env, fail_cell=fail_cell, fail_time=fail_time)
    obs, _ = env.reset(seed=seed)
    done = False

    dt_h = env.dt / 3600.0
    Q_Ah = 0.0
    switches_total = 0
    soc_trace, sw_trace = [], []

    last_bits = env.switch_on.copy()

    while not done:
        # sign convention: >0 discharge, <0 charge (as used previously)
        I_load = env._profile_current(env.t) if hasattr(env, "_profile_current") else 0.0

        # choose action
        if policy == "NoBalancing":
            action = act_nobalancing(env)
        elif policy == "RuleBased":
            action = act_rule_based(env, I_load)
        elif policy in ("PPO", "DQN"):
            action = act_model(model, obs)
        else:
            raise ValueError("Unknown policy")

        obs, reward, done, trunc, info = env.step(action)

        # Usable capacity: integrate discharge only (paper’s convention)
        if isinstance(info, dict) and "I_load" in info:
            I = abs(float(info["I_load"]))
        elif hasattr(env, "I_load"):
            I = abs(float(getattr(env, "I_load")))
        else:
            I = abs(float(env._profile_current(env.t))) if hasattr(env, "_profile_current") else 0.0
        
        # integrate absolute current to ensure capacity accumulation
        Q_Ah += I * dt_h

        bits = env.switch_on.copy()
        switches_total += int(np.sum(bits != last_bits))
        last_bits = bits

        soc_trace.append(env.soc.copy())
        sw_trace.append(bits.copy())

    # metrics at episode end (exclude failed cell from variance)
    if getattr(env, "failed", False):
        mask = np.ones(env.n, dtype=bool)
        mask[env.fail_cell] = False
        var_soc_end = float(np.var(env.soc[mask]))
    else:
        var_soc_end = float(np.var(env.soc))

    return {
        "var_soc": var_soc_end,
        "switch_changes": int(switches_total),
        "usable_capacity_mAh": float(Q_Ah * 1000.0),
        "failed": bool(getattr(env, "failed", False))
    }, np.array(soc_trace), np.array(sw_trace)


# ========== Eval loop, plotting, tables ==========

def evaluate(policy: str, model, episodes: int, seed: int, tag: str,
             fail_cell: int, fail_time: float):
    profile = "charge-rest-discharge"
    results = {profile: []}
    traces = {profile: []}
    for ep in range(episodes):
        ep_res, soc, sw = run_episode(policy, model, profile, seed + ep, fail_cell, fail_time)
        results[profile].append(ep_res)
        if ep == 0:
            traces[profile].append((soc, sw))
    os.makedirs("results_failure", exist_ok=True)
    with open(f"results_failure/metrics_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, traces

def plot_soc_switch(soc, sw, title, out_png, dt=30.0, fail_time=None):
    T = soc.shape[0]
    t = np.arange(T) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)

    for i in range(soc.shape[1]):
        ax1.plot(t, soc[:, i] * 100.0, label=f"Cell {i+1}")
    if fail_time is not None:
        ax1.axvline(fail_time, color='tab:blue', linestyle="--", linewidth=1.5, label="Failure")
    ax1.set_ylabel("SoC (%)")
    ax1.set_title(title)
    ax1.legend(loc="best", ncol=3, fontsize=8)

    for i in range(sw.shape[1]):
        ax2.step(t, sw[:, i], where='post', label=f"Cell {i+1}")
    if fail_time is not None:
        ax2.axvline(fail_time, color='tab:blue', linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Switch (0/1)")
    ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def summarize_metrics(all_metrics, label):
    out = {}
    for profile, eps in all_metrics.items():
        var_m, var_s = ms([e["var_soc"] for e in eps])
        sw_m,  sw_s  = ms([e["switch_changes"] for e in eps])
        cap_m, cap_s = ms([e["usable_capacity_mAh"] for e in eps])
        out[profile] = {"var_mean": var_m, "var_std": var_s,
                        "sw_mean": sw_m, "sw_std": sw_s,
                        "cap_mean": cap_m, "cap_std": cap_s}
    return {label: out}

def write_tables(metrics_by_algo):
    profile = "charge-rest-discharge"
    base = metrics_by_algo["NoBalancing"][profile]["cap_mean"]
    eps = 1e-9

    # Table F1: capacity & improvement vs NoBalancing
    rows1 = []
    for algo in ["RuleBased", "PPO", "DQN"]:
        if algo in metrics_by_algo:
            cap = metrics_by_algo[algo][profile]["cap_mean"]
            imp = 100.0 * (cap - base) / max(base, eps)
            imp_out = imp if base > eps else "N/A"
            rows1.append([algo, cap, imp_out])
    with open("results_failure/tableF1_capacity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm", "Mean Usable Capacity (mAh)", "Mean Improvement (%)"])
        w.writerows(rows1)

    # Table F2: variance/std/switching
    hdr = ["Algorithm", "SoC Variance (mean)", "SoC Std Dev (mean)", "Mean Switch Changes"]
    rows2 = []
    for algo in ["NoBalancing", "RuleBased", "PPO", "DQN"]:
        if algo in metrics_by_algo:
            var_m = metrics_by_algo[algo][profile]["var_mean"]
            std_m = math.sqrt(max(var_m, 0.0))
            sw_m  = metrics_by_algo[algo][profile]["sw_mean"]
            rows2.append([algo, var_m, std_m, sw_m])
    with open("results_failure/tableF2_variance_profile1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows2)


# ========== Main ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", type=str, default=None)
    ap.add_argument("--dqn-ckpt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fail-cell", type=int, default=2)        # 0-based index
    ap.add_argument("--fail-time", type=float, default=4200.0) # seconds
    args = ap.parse_args()

    os.makedirs("results_failure", exist_ok=True)
    os.makedirs("plots_failure", exist_ok=True)

    metrics_by_algo = {}

    # NoBalancing
    nb_metrics, nb_traces = evaluate("NoBalancing", None, args.episodes, args.seed, "nobalancing", args.fail_cell, args.fail_time)
    metrics_by_algo.update(summarize_metrics(nb_metrics, "NoBalancing"))
    for profile, lst in nb_traces.items():
        if lst:
            soc, sw = lst[0]
            plot_soc_switch(soc, sw, f"NoBalancing — {profile} — fail c{args.fail_cell+1} @ {args.fail_time:.1f}s",
                            f"plots_failure/nobalancing_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                            fail_time=args.fail_time)

    # RuleBased
    rb_metrics, rb_traces = evaluate("RuleBased", None, args.episodes, args.seed, "rulebased", args.fail_cell, args.fail_time)
    metrics_by_algo.update(summarize_metrics(rb_metrics, "RuleBased"))
    for profile, lst in rb_traces.items():
        if lst:
            soc, sw = lst[0]
            plot_soc_switch(soc, sw, f"RuleBased — {profile} — fail c{args.fail_cell+1} @ {args.fail_time:.1f}s",
                            f"plots_failure/rulebased_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                            fail_time=args.fail_time)

    # PPO
    if args.ppo_ckpt:
        ppo_model = PPO.load(args.ppo_ckpt, custom_objects={
            "lr_schedule": (lambda *_, **__: 0.0),
            "clip_range":  (lambda *_, **__: 0.2),
        })
        ppo_metrics, ppo_traces = evaluate("PPO", ppo_model, args.episodes, args.seed, "ppo", args.fail_cell, args.fail_time)
        metrics_by_algo.update(summarize_metrics(ppo_metrics, "PPO"))
        for profile, lst in ppo_traces.items():
            if lst:
                soc, sw = lst[0]
                plot_soc_switch(soc, sw, f"PPO — {profile} — fail c{args.fail_cell+1} @ {args.fail_time:.1f}s",
                                f"plots_failure/ppo_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                                fail_time=args.fail_time)
    else:
        print("[warn] --ppo-ckpt not provided; skipping PPO.")

    # DQN
    if args.dqn_ckpt:
        dqn_model = DQN.load(args.dqn_ckpt, custom_objects={
            "lr_schedule":          (lambda *_, **__: 0.0),
            "exploration_schedule": (lambda *_, **__: 0.0),
        })
        dqn_metrics, dqn_traces = evaluate("DQN", dqn_model, args.episodes, args.seed, "dqn", args.fail_cell, args.fail_time)
        metrics_by_algo.update(summarize_metrics(dqn_metrics, "DQN"))
        for profile, lst in dqn_traces.items():
            if lst:
                soc, sw = lst[0]
                plot_soc_switch(soc, sw, f"DQN — {profile} — fail c{args.fail_cell+1} @ {args.fail_time:.1f}s",
                                f"plots_failure/dqn_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                                fail_time=args.fail_time)
    else:
        print("[warn] --dqn-ckpt not provided; skipping DQN.")

    write_tables(metrics_by_algo)
    with open("results_failure/paper_summary_failure.json", "w") as f:
        json.dump(metrics_by_algo, f, indent=2)
    print("[✓] Done. Tables -> results_failure/, Plots -> plots_failure/")


if __name__ == "__main__":
    main()
