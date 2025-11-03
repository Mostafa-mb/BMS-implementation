import os, argparse, csv, json, math
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from gym_bms import BatteryPackEnv
from stable_baselines3 import PPO, DQN


# ========== Helper Functions ==========

def force_bit(action: int, bit_index: int, value: int) -> int:
    """Force bit at index to given value (0 or 1)."""
    mask = 1 << bit_index
    action &= ~mask
    if value:
        action |= mask
    return action


def ms(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


# ========== Policy Definitions ==========

def act_nobalancing(env: BatteryPackEnv) -> int:
    return 0


def act_rule_based(env: BatteryPackEnv, I_load: float) -> int:
    tol = 0.01
    bits = np.zeros(env.n, dtype=np.int32)
    mean_soc = float(np.mean(env.soc))
    if I_load > 0:  # discharge
        for i in range(env.n):
            if env.soc[i] > mean_soc + tol:
                bits[i] = 1
    elif I_load < 0:  # charge
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


# ========== Core Episode ==========

def run_episode(policy: str, model, profile: str, seed: int,
                fail_cell: int, fail_time: float):
    env = BatteryPackEnv(seed=seed, profile=profile)
    obs, _ = env.reset(seed=seed)
    done = False

    dt_h = env.dt / 3600.0
    Q_Ah = 0.0
    switches_total = 0
    soc_trace, sw_trace = [], []

    last_bits = env.switch_on.copy()
    failed = False
    fail_step_seen = False
    last_soc = env.soc.copy()

    while not done:
        I_load = env._profile_current(env.t)

        # Choose policy
        if policy == "NoBalancing":
            action = act_nobalancing(env)
        elif policy == "RuleBased":
            action = act_rule_based(env, I_load)
        elif policy in ("PPO", "DQN"):
            action = act_model(model, obs)
        else:
            raise ValueError("Unknown policy")

        # Check failure
        if (not failed) and (env.t >= fail_time):
            failed = True

        # Disable failed cell
        if failed:
            action = force_bit(action, fail_cell, 0)

        obs, reward, done, trunc, info = env.step(action)

        I = info.get("I_load", 0.0)
        if I > 0:
            Q_Ah += I * dt_h

        bits = env.switch_on.copy()

        if failed:
            bits_for_count = bits.copy()
            bits_for_count[fail_cell] = last_bits[fail_cell]
        else:
            bits_for_count = bits

        switches_total += int(np.sum(bits_for_count != last_bits))
        last_bits = bits_for_count

        soc_plot = env.soc.copy()
        sw_plot = bits.copy()
        if failed:
            if not fail_step_seen:
                last_soc = soc_plot.copy()
                fail_step_seen = True
            soc_plot[fail_cell] = last_soc[fail_cell]  # flatten
            sw_plot[fail_cell] = 0  # mask off
        soc_trace.append(soc_plot)
        sw_trace.append(sw_plot)

    soc_end = env.soc.copy()
    soc_masked = np.delete(soc_end, fail_cell) if failed else soc_end
    var_soc_end = float(np.var(soc_masked))

    return {
        "var_soc": var_soc_end,
        "switch_changes": int(switches_total),
        "usable_capacity_mAh": float(Q_Ah * 1000.0),
        "failed": bool(failed)
    }, np.array(soc_trace), np.array(sw_trace)


# ========== Evaluation Loop ==========

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


# ========== Plotting ==========

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


# ========== Summaries & Tables ==========

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

    # Table F1
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

    # Table F2
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
    ap.add_argument("--fail-cell", type=int, default=2)
    ap.add_argument("--fail-time", type=float, default=4200.0)
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
