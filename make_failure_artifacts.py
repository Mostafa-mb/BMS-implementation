
import os, argparse, csv, json, math
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from gym_bms import BatteryPackEnv
from stable_baselines3 import PPO, DQN

def act_nobalancing(env: BatteryPackEnv) -> int:
    return 0

def act_rule_based(env: BatteryPackEnv, I_load: float) -> int:
    tol = 0.01
    bits = np.zeros(env.n, dtype=np.int32)
    mean_soc = float(np.mean(env.soc))
    if I_load > 0:
        for i in range(env.n):
            if env.soc[i] > mean_soc + tol:
                bits[i] = 1
    elif I_load < 0:
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

def force_bit(action: int, bit_index: int, value: int) -> int:
    mask = 1 << bit_index
    action &= ~mask
    if value:
        action |= mask
    return action

def run_episode(policy: str, model, profile: str, seed: int, fail_cell: int, fail_time: float):
    env = BatteryPackEnv(seed=seed, profile=profile)
    obs, _ = env.reset(seed=seed)
    done = False
    Q_Ah = 0.0
    dt_h = env.dt / 3600.0
    switches_total = 0
    soc_trace = []
    sw_trace = []
    last_bits = env.switch_on.copy()
    failed = False

    while not done:
        I_load = env._profile_current(env.t)
        if policy == "NoBalancing":
            action = act_nobalancing(env)
        elif policy == "RuleBased":
            action = act_rule_based(env, I_load)
        elif policy in ["PPO","DQN"]:
            action = act_model(model, obs)
        else:
            raise ValueError("Unknown policy")

        if (not failed) and (env.t >= fail_time):
            failed = True
        if failed:
            action = force_bit(action, fail_cell, 0)

        obs, reward, done, trunc, info = env.step(action)

        I = info.get("I_load", 0.0)
        if I > 0:
            Q_Ah += I * dt_h

        bits = env.switch_on.copy()
        if failed:
            bits[fail_cell] = last_bits[fail_cell]
        switches_total += int(np.sum(bits != last_bits))
        last_bits = bits

        soc_trace.append(env.soc.copy())
        sw_trace.append(env.switch_on.copy())

    soc_end = env.soc.copy()
    if failed:
        soc_masked = np.delete(soc_end, fail_cell)
    else:
        soc_masked = soc_end
    var_soc_end = float(np.var(soc_masked))

    ep = {
        "var_soc": var_soc_end,
        "switch_changes": int(switches_total),
        "usable_capacity_mAh": float(Q_Ah * 1000.0),
        "failed": bool(failed)
    }
    return ep, np.array(soc_trace), np.array(sw_trace)

def evaluate(policy: str, model, episodes: int, seed: int, tag: str, fail_cell: int, fail_time: float):
    results = {"charge-rest-discharge": []}
    traces = {"charge-rest-discharge": []}
    profile = "charge-rest-discharge"
    for ep in range(episodes):
        ep_res, soc, sw = run_episode(policy, model, profile, seed + ep, fail_cell, fail_time)
        results[profile].append(ep_res)
        if ep == 0:
            traces[profile].append((soc, sw))
    os.makedirs("results_failure", exist_ok=True)
    with open(f"results_failure/metrics_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, traces

def plot_soc_switch(soc: np.ndarray, sw: np.ndarray, title: str, out_png: str, dt=30.0, fail_time=None):
    t = np.arange(soc.shape[0]) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    for i in range(soc.shape[1]):
        ax1.plot(t, soc[:,i]*100.0, label=f"Cell {i+1}")
    if fail_time is not None:
        ax1.axvline(fail_time, linestyle="--")
    ax1.set_ylabel("SoC (%)"); ax1.set_title(title); ax1.legend(loc="best", ncol=3, fontsize=8)
    for i in range(sw.shape[1]):
        ax2.step(t, sw[:,i], where='post')
    if fail_time is not None:
        ax2.axvline(fail_time, linestyle="--")
    ax2.set_ylabel("Switch (0/1)"); ax2.set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def summarize_metrics(all_metrics: dict, label: str) -> dict:
    def mean_std(vals):
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std())
    out = {}
    for profile, eps in all_metrics.items():
        var_m, var_s = mean_std([e["var_soc"] for e in eps])
        sw_m,  sw_s  = mean_std([e["switch_changes"] for e in eps])
        cap_m, cap_s = mean_std([e["usable_capacity_mAh"] for e in eps])
        out[profile] = {"var_mean": var_m, "var_std": var_s, "sw_mean": sw_m, "sw_std": sw_s, "cap_mean": cap_m, "cap_std": cap_s}
    return {label: out}

def write_tables(metrics_by_algo: dict):
    # Capacity table (profile 1 only: charge–rest–discharge)
    profile = "charge-rest-discharge"
    base = metrics_by_algo["NoBalancing"][profile]["cap_mean"]
    eps = 1e-9  # guard

    rows1 = []
    for algo in ["RuleBased", "PPO", "DQN"]:
        if algo in metrics_by_algo:
            cap = metrics_by_algo[algo][profile]["cap_mean"]
            if base > eps:
                imp = 100.0 * (cap - base) / base
            else:
                imp = "N/A"  # baseline capacity is zero → no % comparison
            rows1.append([algo, cap, imp])
        else:
            rows1.append([algo, "", ""])

    os.makedirs("results_failure", exist_ok=True)
    with open("results_failure/tableF1_capacity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm", "Mean Usable Capacity (mAh)", "Mean Improvement (%)"])
        w.writerows(rows1)

    # Variance / std / switching (same profile)
    hdr = ["Algorithm", "SoC Variance (mean)", "SoC Std Dev (mean)", "Mean Switch Changes"]
    rows = []
    for algo in ["NoBalancing", "RuleBased", "PPO", "DQN"]:
        if algo in metrics_by_algo:
            var_m = metrics_by_algo[algo][profile]["var_mean"]
            std_m = math.sqrt(max(var_m, 0.0))
            sw_m  = metrics_by_algo[algo][profile]["sw_mean"]
            rows.append([algo, var_m, std_m, sw_m])
        else:
            rows.append([algo, "", "", ""])
    with open("results_failure/tableF2_variance_profile1.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", type=str, default=None)
    ap.add_argument("--dqn-ckpt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fail-cell", type=int, default=0)
    ap.add_argument("--fail-time", type=float, default=900.0)
    args = ap.parse_args()

    os.makedirs("results_failure", exist_ok=True)
    os.makedirs("plots_failure", exist_ok=True)

    metrics_by_algo = {}

    nb_metrics, nb_traces = evaluate("NoBalancing", None, args.episodes, args.seed, "nobalancing", args.fail_cell, args.fail_time)
    metrics_by_algo.update(summarize_metrics(nb_metrics, "NoBalancing"))
    for profile, lst in nb_traces.items():
        if lst:
            soc, sw = lst[0]
            plot_soc_switch(soc, sw, f"NoBalancing — {profile} — fail c{args.fail_cell} @ {args.fail_time}s",
                            f"plots_failure/nobalancing_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                            fail_time=args.fail_time)

    rb_metrics, rb_traces = evaluate("RuleBased", None, args.episodes, args.seed, "rulebased", args.fail_cell, args.fail_time)
    metrics_by_algo.update(summarize_metrics(rb_metrics, "RuleBased"))
    for profile, lst in rb_traces.items():
        if lst:
            soc, sw = lst[0]
            plot_soc_switch(soc, sw, f"RuleBased — {profile} — fail c{args.fail_cell} @ {args.fail_time}s",
                            f"plots_failure/rulebased_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                            fail_time=args.fail_time)

    if args.ppo_ckpt:
        custom = {
        "lr_schedule": lambda *_: 0.0,   # fixed lr after load (or set to your lr)
        "clip_range":  lambda *_: 0.2,   # fixed clip after load
        }
        ppo_model = PPO.load(args.ppo_ckpt, custom_objects=custom)
        ppo_metrics, ppo_traces = evaluate("PPO", ppo_model, args.episodes, args.seed, "ppo", args.fail_cell, args.fail_time)
        metrics_by_algo.update(summarize_metrics(ppo_metrics, "PPO"))
        for profile, lst in ppo_traces.items():
            if lst:
                soc, sw = lst[0]
                plot_soc_switch(soc, sw, f"PPO — {profile} — fail c{args.fail_cell} @ {args.fail_time}s",
                                f"plots_failure/ppo_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                                fail_time=args.fail_time)
    else:
        print("[warn] --ppo-ckpt not provided; skipping PPO.")

    if args.dqn_ckpt:
        custom = {
        "lr_schedule":            lambda *_: 0.0,  # fixed lr after load
        "exploration_schedule":   lambda *_: 0.0,  # use current exploration settings
        }
        dqn_model = DQN.load(args.dqn_ckpt, custom_objects=custom)
        dqn_metrics, dqn_traces = evaluate("DQN", dqn_model, args.episodes, args.seed, "dqn", args.fail_cell, args.fail_time)
        metrics_by_algo.update(summarize_metrics(dqn_metrics, "DQN"))
        for profile, lst in dqn_traces.items():
            if lst:
                soc, sw = lst[0]
                plot_soc_switch(soc, sw, f"DQN — {profile} — fail c{args.fail_cell} @ {args.fail_time}s",
                                f"plots_failure/dqn_{profile}_FAILc{args.fail_cell}_t{args.fail_time}_soc_switch.png",
                                fail_time=args.fail_time)
    else:
        print("[warn] --dqn-ckpt not provided; skipping DQN.")

    write_tables(metrics_by_algo)
    with open("results_failure/paper_summary_failure.json","w") as f:
        json.dump(metrics_by_algo, f, indent=2)
    print("[✓] Done. Tables -> results_failure/, Plots -> plots_failure/")

if __name__ == "__main__":
    main()
