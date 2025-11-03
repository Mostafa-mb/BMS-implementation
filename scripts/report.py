import os, glob, re, argparse, json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from gym_bms import BatteryPackEnv

# --------- Utilities ----------
def latest_checkpoint(ckpt_dir: str):
    """Pick newest rl_model_*_steps.zip, else best_model/final_model if present."""
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "rl_model_*_steps.zip"))
    def steps(p):
        m = re.search(r"rl_model_(\d+)_steps\\.zip$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    cands.sort(key=steps, reverse=True)
    if cands:
        return cands[0]
    for name in ["best_model.zip", "final_model.zip"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    return None

def load_model(algo: str, ckpt_path: str, env):
    if algo == "dqn":
        return DQN.load(ckpt_path, env=env)
    else:
        return PPO.load(ckpt_path, env=env)

def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

# ---------- Evaluation ----------
def eval_once(model, profile: str, seed: int, ep_idx: int):
    env = BatteryPackEnv(seed=seed + ep_idx, profile=profile)
    obs, _ = env.reset(seed=seed + ep_idx)
    done = False
    switches_total = 0
    working_steps = 0
    total_steps = 0
    soc_trace = []
    sw_trace  = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        switches_total += info.get("switches", 0)
        # working if all cells within 10–90% SoC window
        if (env.soc >= env.soc_min).all() and (env.soc <= env.soc_max).all():
            working_steps += 1
        total_steps += 1
        soc_trace.append(env.soc.copy())
        sw_trace.append(env.switch_on.copy())
    soc_trace = np.array(soc_trace)   # T x 5
    sw_trace  = np.array(sw_trace)    # T x 5
    var_soc   = float(np.var(env.soc))
    working_ratio = float(working_steps) / max(1, total_steps)
    return {
        "var_soc": var_soc,
        "switch_changes": int(switches_total),
        "working_ratio": working_ratio,
    }, soc_trace, sw_trace

def eval_model(algo: str, ckpt_path: str, seed: int, episodes: int):
    metrics = {}
    traces  = {}  # (profile -> list of (soc, switch))
    for profile in ["discharge-rest-charge", "charge-rest-discharge"]:
        env = BatteryPackEnv(seed=seed, profile=profile)
        model = load_model(algo, ckpt_path, env)
        per_ep = []
        ep_traces = []
        for ep in range(episodes):
            m, soc, sw = eval_once(model, profile, seed, ep)
            per_ep.append(m)
            ep_traces.append((soc, sw))
        metrics[profile] = per_ep
        traces[profile]  = ep_traces
    return metrics, traces

# ---------- Plotting ----------
def plot_soc(arr, title, out_png, dt=30.0):
    T = arr.shape[0]
    t = np.arange(T) * dt
    for i in range(arr.shape[1]):
        plt.plot(t, arr[:, i] * 100.0, label=f"Cell {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("SoC (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_switch(arr, title, out_png, dt=30.0):
    T = arr.shape[0]
    t = np.arange(T) * dt
    for i in range(arr.shape[1]):
        plt.step(t, arr[:, i], where="post", label=f"Cell {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Switch (0/1)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------- Summaries ----------
def summarize_to_csv(all_results: dict, out_csv: str):
    """
    all_results structure:
      { algo: { profile: [ {var_soc, switch_changes, working_ratio}, ... ] } }
    Writes a CSV with mean ± std across episodes.
    """
    import csv
    rows = []
    for algo, profs in all_results.items():
        for profile, eps in profs.items():
            if not eps:
                continue
            var_list = [e["var_soc"] for e in eps]
            sw_list  = [e["switch_changes"] for e in eps]
            wr_list  = [e["working_ratio"] for e in eps]
            def m_s(a):
                arr = np.array(a, dtype=float)
                return float(np.mean(arr)), float(np.std(arr))
            m_var, s_var = m_s(var_list)
            m_sw,  s_sw  = m_s(sw_list)
            m_wr,  s_wr  = m_s(wr_list)
            rows.append([algo, profile, m_var, s_var, m_sw, s_sw, m_wr, s_wr, len(eps)])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm","Profile","Var(SoC)_mean","Var(SoC)_std",
                    "Switches_mean","Switches_std","Working_ratio_mean","Working_ratio_std","Episodes"])
        w.writerows(rows)
    return out_csv

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["both","ppo","dqn"], default="both")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dqn-ckpt", type=str, default=None)
    parser.add_argument("--ppo-ckpt", type=str, default=None)
    parser.add_argument("--ckpt-root", type=str, default="checkpoints")
    parser.add_argument("--dqn-seed", type=int, default=42)
    parser.add_argument("--ppo-seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    algos = []
    if args.algo in ("both","dqn"): algos.append("dqn")
    if args.algo in ("both","ppo"): algos.append("ppo")

    ckpt_map = {}
    for algo in algos:
        # Allow override; otherwise find latest in checkpoints/{algo}_seed{seed}
        override = args.dqn_ckpt if algo=="dqn" else args.ppo_ckpt
        if override and os.path.exists(override):
            ckpt_map[algo] = override
        else:
            seed_used = args.dqn_seed if algo=="dqn" else args.ppo_seed
            ckpt_dir = os.path.join(args.ckpt_root, f"{algo}_seed{seed_used}")
            ckpt_path = latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found for {algo} in {ckpt_dir}. Train first.")
            ckpt_map[algo] = ckpt_path
        print(f"[report] Using {algo.upper()} checkpoint: {ckpt_map[algo]}")

    all_results = {}
    for algo in algos:
        seed_used = args.dqn_seed if algo=="dqn" else args.ppo_seed
        metrics, traces = eval_model(algo, ckpt_map[algo], seed=seed_used, episodes=args.episodes)
        all_results[algo] = metrics
        # Save raw metrics JSON for record-keeping
        with open(f"results/metrics_{algo}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # Plot first episode’s traces for each profile
        for profile, ep_list in traces.items():
            if not ep_list:
                continue
            soc0, sw0 = ep_list[0]
            plot_soc(soc0, f"SoC over time — {algo.upper()} — {profile}", f"plots/{algo}_{profile}_soc.png")
            plot_switch(sw0, f"Switching — {algo.upper()} — {profile}", f"plots/{algo}_{profile}_switch.png")

    out_csv = summarize_to_csv(all_results, "results/summary.csv")
    print(f"[report] Wrote {out_csv}")
    print("[report] Plots saved in plots/ and run-level metrics in results/metrics_*.json")

if __name__ == "__main__":
    main()