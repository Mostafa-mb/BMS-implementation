
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

def run_episode(policy: str, model, profile: str, seed: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
    env = BatteryPackEnv(seed=seed, profile=profile)
    obs, _ = env.reset(seed=seed)
    done = False
    Q_Ah = 0.0
    dt_h = env.dt / 3600.0
    switches_total = 0
    soc_trace = []
    sw_trace = []
    last_bits = env.switch_on.copy()
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
        obs, reward, done, trunc, info = env.step(action)
        I = info.get("I_load", 0.0)
        if I > 0:
            Q_Ah += I * dt_h
        bits = env.switch_on.copy()
        switches_total += int(np.sum(bits != last_bits))
        last_bits = bits
        soc_trace.append(env.soc.copy())
        sw_trace.append(bits.copy())
    var_soc_end = float(np.var(env.soc))
    ep = {
        "var_soc": var_soc_end,
        "switch_changes": int(switches_total),
        "usable_capacity_mAh": float(Q_Ah * 1000.0),
    }
    return ep, np.array(soc_trace), np.array(sw_trace)

def evaluate(policy: str, model, episodes: int, seed: int, tag: str):
    results = {"discharge-rest-charge": [], "charge-rest-discharge": []}
    traces = {}
    for profile in ["discharge-rest-charge", "charge-rest-discharge"]:
        traces[profile] = []
        for ep in range(episodes):
            ep_res, soc, sw = run_episode(policy, model, profile, seed + ep)
            results[profile].append(ep_res)
            if ep == 0:
                traces[profile].append((soc, sw))
    os.makedirs("results", exist_ok=True)
    with open(f"results/metrics_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, traces

def plot_soc_switch(soc: np.ndarray, sw: np.ndarray, title: str, out_png: str, dt=30.0):
    t = np.arange(soc.shape[0]) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    for i in range(soc.shape[1]):
        ax1.plot(t, soc[:,i]*100.0, label=f"Cell {i+1}")
    ax1.set_ylabel("SoC (%)"); ax1.set_title(title); ax1.legend(loc="best", ncol=3, fontsize=8)
    for i in range(sw.shape[1]):
        ax2.step(t, sw[:,i], where='post', label=f"Cell {i+1}")
    ax2.set_ylabel("Switch (0/1)"); ax2.set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def summarize_metrics(all_metrics: dict, label: str) -> dict:
    def ms(a):
        arr = np.array(a, dtype=float)
        return float(arr.mean()), float(arr.std())
    out = {}
    for profile, eps in all_metrics.items():
        var_m, var_s = ms([e["var_soc"] for e in eps])
        sw_m,  sw_s  = ms([e["switch_changes"] for e in eps])
        cap_m, cap_s = ms([e["usable_capacity_mAh"] for e in eps])
        out[profile] = {"var_mean": var_m, "var_std": var_s, "sw_mean": sw_m, "sw_std": sw_s, "cap_mean": cap_m, "cap_std": cap_s}
    return {label: out}

def write_tables(metrics_by_algo: dict):
    base = metrics_by_algo["NoBalancing"]["discharge-rest-charge"]["cap_mean"]
    rows1 = []
    for algo in ["PPO","TRPO","DQN"]:
        if algo in metrics_by_algo:
            cap = metrics_by_algo[algo]["discharge-rest-charge"]["cap_mean"]
            imp = 100.0*(cap - base)/base
            rows1.append([algo, cap, imp])
        else:
            rows1.append([algo, "", ""])
    os.makedirs("results", exist_ok=True)
    with open("results/table1_capacity.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Algorithm","Mean Usable Capacity (mAh)","Mean Improvement (%)"]); w.writerows(rows1)

    def write_var_table(profile, out_csv):
        hdr = ["Algorithm","SoC Variance (mean)","SoC Std Dev (mean)","Mean Switch Changes"]
        rows = []
        for algo in ["NoBalancing","RuleBased","PPO","TRPO","DQN"]:
            if algo in metrics_by_algo:
                var_m = metrics_by_algo[algo][profile]["var_mean"]
                std_m = math.sqrt(max(var_m, 0.0))
                sw_m  = metrics_by_algo[algo][profile]["sw_mean"]
                rows.append([algo, var_m, std_m, sw_m])
            else:
                rows.append([algo, "", "", ""])
        with open(out_csv,"w",newline="") as f:
            w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
    write_var_table("discharge-rest-charge","results/table2_variance_profile1.csv")
    write_var_table("charge-rest-discharge","results/table3_variance_profile2.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", type=str, default=None)
    ap.add_argument("--dqn-ckpt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    metrics_by_algo = {}

    nb_metrics, nb_traces = evaluate("NoBalancing", None, args.episodes, args.seed, "nobalancing")
    metrics_by_algo.update(summarize_metrics(nb_metrics, "NoBalancing"))
    for profile, lst in nb_traces.items():
        if lst:
            soc, sw = lst[0]
            plot_soc_switch(soc, sw, f"NoBalancing — {profile}", f"plots/nobalancing_{profile}_soc_switch.png")

    rb_metrics, rb_traces = evaluate("RuleBased", None, args.episodes, args.seed, "rulebased")
    metrics_by_algo.update(summarize_metrics(rb_metrics, "RuleBased"))
    for profile, lst in rb_traces.items():
        if lst:
            soc, sw = lst[0]
            plot_soc_switch(soc, sw, f"RuleBased — {profile}", f"plots/rulebased_{profile}_soc_switch.png")

    if args.ppo_ckpt:
        ppo_model = PPO.load(args.ppo_ckpt)
        ppo_metrics, ppo_traces = evaluate("PPO", ppo_model, args.episodes, args.seed, "ppo")
        metrics_by_algo.update(summarize_metrics(ppo_metrics, "PPO"))
        for profile, lst in ppo_traces.items():
            if lst:
                soc, sw = lst[0]
                plot_soc_switch(soc, sw, f"PPO — {profile}", f"plots/ppo_{profile}_soc_switch.png")
    else:
        print("[warn] --ppo-ckpt not provided; skipping PPO.")

    if args.dqn_ckpt:
        dqn_model = DQN.load(args.dqn_ckpt)
        dqn_metrics, dqn_traces = evaluate("DQN", dqn_model, args.episodes, args.seed, "dqn")
        metrics_by_algo.update(summarize_metrics(dqn_metrics, "DQN"))
        for profile, lst in dqn_traces.items():
            if lst:
                soc, sw = lst[0]
                plot_soc_switch(soc, sw, f"DQN — {profile}", f"plots/dqn_{profile}_soc_switch.png")
    else:
        print("[warn] --dqn-ckpt not provided; skipping DQN.")

    write_tables(metrics_by_algo)
    with open("results/paper_summary.json","w") as f:
        json.dump(metrics_by_algo, f, indent=2)
    print("[✓] Done. Tables -> results/, Plots -> plots/")

if __name__ == "__main__":
    main()
