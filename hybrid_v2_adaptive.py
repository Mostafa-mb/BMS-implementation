#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid-V2 (Adaptive PPO+DQN Arbiter)
- Softmax-gated ensemble (no retraining)
- Evaluates PPO, DQN *and* Hybrid in one run
- Robust capacity integration and switch counting
- Produces JSON + CSV + comparison chart

Usage (PowerShell example):
python .\hybrid_v2_adaptive.py `
  --ppo-ckpt "checkpoints\ppo_seed42\final_model.zip" `
  --dqn-ckpt "checkpoints\dqn_seed42\final_model.zip" `
  --episodes 10 --seed 123 `
  --profile "charge-rest-discharge" `
  --k 40 --theta 0.0045 --max-switch-budget 60 `
  --decay 0.98 --window 15 --min-dqn-span 6 --min-ppo-span 6
"""

import os, json, math, argparse, random
from collections import deque
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from gym_bms import BatteryPackEnv


# ---------------------------- Utilities ----------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def soc_variance(vec: np.ndarray) -> float:
    v = np.array(vec, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.var(v)) if v.size else 0.0

def count_switch_changes(prev_bits: np.ndarray, bits: np.ndarray) -> int:
    if prev_bits is None:
        return 0
    return int(np.sum(prev_bits != bits))

def get_pack_current(env, info) -> float:
    # Robust across env variants: abs(current) to avoid sign assumptions
    if isinstance(info, dict) and ("I_load" in info):
        return abs(float(info["I_load"]))
    if hasattr(env, "I_load"):
        return abs(float(getattr(env, "I_load")))
    if hasattr(env, "_profile_current"):
        try:
            return abs(float(env._profile_current(env.t)))
        except Exception:
            return 0.0
    return 0.0


# ---------------------------- Baseline Policies ----------------------------

def act_ppo(model: PPO, obs) -> int:
    a, _ = model.predict(obs, deterministic=True)
    return int(a)

def act_dqn(model: DQN, obs) -> int:
    a, _ = model.predict(obs, deterministic=True)
    return int(a)


# ---------------------------- Hybrid-V2 Arbiter ----------------------------

class AdaptiveHybrid:
    """
    Adaptive softmax gating between PPO and DQN:
    - Computes running mean of SoC variance over last `window` steps
    - Urgency U_t ~ (var_t - mean_recent) / (var_t + eps)
    - P(DQN) = sigmoid(k * (U_t - theta))
    - P(DQN) further scaled by remaining switch budget fraction
    - Enforces min-span (in steps) to avoid ping-ponging (commitment)
    - Decays allowed budget when variance remains low for many steps
    """
    def __init__(
        self,
        ppo_path: str,
        dqn_path: str,
        k: float = 40.0,
        theta: float = 0.0045,
        max_switch_budget: int = 60,
        decay: float = 0.98,
        window: int = 15,
        min_dqn_span: int = 6,
        min_ppo_span: int = 6,
        seed: int = 123
    ):
        self.ppo = PPO.load(ppo_path, custom_objects={
            "lr_schedule": (lambda *_, **__: 0.0),
            "clip_range":  (lambda *_, **__: 0.2),
        })
        self.dqn = DQN.load(dqn_path, custom_objects={
            "lr_schedule":          (lambda *_, **__: 0.0),
            "exploration_schedule": (lambda *_, **__: 0.0),
        })
        self.k = float(k)
        self.theta = float(theta)
        self.max_switch_budget = int(max_switch_budget)
        self.decay = float(decay)
        self.window = int(window)
        self.min_dqn_span = int(min_dqn_span)
        self.min_ppo_span = int(min_ppo_span)
        self.eps = 1e-9

        self._reset_internal(seed)

    def _reset_internal(self, seed: int):
        set_all_seeds(seed)
        self.var_hist = deque(maxlen=self.window)
        self.allowed_switch_budget = self.max_switch_budget
        self.switches_used = 0
        self.last_bits = None
        self.current_expert = "PPO"  # start calm
        self.commit_steps_left = self.min_ppo_span
        self.steps_ppo = 0
        self.steps_dqn = 0

    def reset_episode(self, seed: int):
        self._reset_internal(seed)

    def _p_dqn(self, var_now: float) -> float:
        mean_recent = np.mean(self.var_hist) if len(self.var_hist) > 0 else var_now
        urgency = (var_now - mean_recent) / max(var_now, self.eps)
        # Soft trigger:
        p = 1.0 / (1.0 + math.exp(-self.k * (urgency - self.theta)))
        # Scale by remaining budget fraction (clipped 0..1)
        remain_frac = max(0.0, min(1.0, (self.allowed_switch_budget - self.switches_used) / max(self.allowed_switch_budget, 1)))
        return float(p * remain_frac)

    def act(self, obs, env) -> Tuple[int, str]:
        var_now = soc_variance(env.soc)
        self.var_hist.append(var_now)

        # Decay budget slowly when stable (var below theta)
        if var_now < self.theta:
            self.allowed_switch_budget = max(1.0, self.allowed_switch_budget * self.decay)

        # Decide expert
        if self.commit_steps_left > 0:
            # honor current commitment
            expert = self.current_expert
            self.commit_steps_left -= 1
        else:
            # probabilistic gating
            pdqn = self._p_dqn(var_now)

            # Draw decision with reproducible RNG
            r = np.random.rand()
            chosen = "DQN" if r < pdqn else "PPO"

            # If budget exhausted, force PPO
            if self.switches_used >= self.allowed_switch_budget:
                chosen = "PPO"

            # Commit for min span to avoid chattering
            if chosen == "DQN":
                self.current_expert = "DQN"
                self.commit_steps_left = self.min_dqn_span
            else:
                self.current_expert = "PPO"
                self.commit_steps_left = self.min_ppo_span
            expert = self.current_expert

        # Query the chosen expert
        if expert == "DQN":
            a = act_dqn(self.dqn, obs)
        else:
            a = act_ppo(self.ppo, obs)

        return int(a), expert

    def update_switches(self, env_bits: np.ndarray):
        self.switches_used += count_switch_changes(self.last_bits, env_bits)
        self.last_bits = env_bits.copy() if env_bits is not None else None

    def log_step_expert(self, expert: str):
        if expert == "DQN":
            self.steps_dqn += 1
        else:
            self.steps_ppo += 1


# ---------------------------- Evaluation Routines ----------------------------

def run_controller(env: BatteryPackEnv, policy_fn, seed: int) -> Dict:
    """
    policy_fn(env, obs) -> (action:int, tag:str)
    """
    obs, _ = env.reset(seed=seed)
    dt_h = env.dt / 3600.0
    Q_Ah = 0.0

    done = False
    last_bits = None
    flips = 0
    info_last = {}

    while not done:
        a, tag = policy_fn(env, obs)  # (action, tag)
        obs, reward, done, trunc, info = env.step(a)

        I = get_pack_current(env, info)
        Q_Ah += I * dt_h

        flips += count_switch_changes(last_bits, env.switch_on)
        last_bits = env.switch_on.copy()
        info_last = info

    return {
        "capacity_mAh": float(Q_Ah * 1000.0),
        "soc_variance": soc_variance(env.soc),
        "switches": int(flips)
    }


def run_hybrid(env: BatteryPackEnv, hybrid: AdaptiveHybrid, seed: int) -> Dict:
    obs, _ = env.reset(seed=seed)
    dt_h = env.dt / 3600.0
    Q_Ah = 0.0

    done = False
    hybrid.reset_episode(seed)
    info_last = {}

    while not done:
        a, expert = hybrid.act(obs, env)
        obs, reward, done, trunc, info = env.step(a)

        I = get_pack_current(env, info)
        Q_Ah += I * dt_h

        # switches
        hybrid.update_switches(env.switch_on)
        # expert usage log
        hybrid.log_step_expert(expert)
        info_last = info

    total_steps = hybrid.steps_ppo + hybrid.steps_dqn + 1e-9
    return {
        "capacity_mAh": float(Q_Ah * 1000.0),
        "soc_variance": soc_variance(env.soc),
        "switches": int(hybrid.switches_used),
        "ppo_step_share": float(hybrid.steps_ppo / total_steps),
        "dqn_step_share": float(hybrid.steps_dqn / total_steps),
        "allowed_switch_budget_effective": float(hybrid.allowed_switch_budget)
    }


def eval_all(ppo_path: str, dqn_path: str, episodes: int, seed: int,
             profile: str, k: float, theta: float, max_switch_budget: int,
             decay: float, window: int, min_dqn_span: int, min_ppo_span: int) -> Dict:
    """
    Runs PPO baseline, DQN baseline, and Hybrid-V2 on the same seeds + profile.
    Returns aggregated metrics (mean/std) and per-episode arrays.
    """
    set_all_seeds(seed)
    out = {
        "params": {
            "profile": profile, "episodes": episodes, "seed": seed,
            "k": k, "theta": theta, "max_switch_budget": max_switch_budget,
            "decay": decay, "window": window,
            "min_dqn_span": min_dqn_span, "min_ppo_span": min_ppo_span
        },
        "baselines": {},
        "hybrid": {}
    }

    # Load models once
    ppo_model = PPO.load(ppo_path, custom_objects={
        "lr_schedule": (lambda *_, **__: 0.0),
        "clip_range":  (lambda *_, **__: 0.2),
    })
    dqn_model = DQN.load(dqn_path, custom_objects={
        "lr_schedule":          (lambda *_, **__: 0.0),
        "exploration_schedule": (lambda *_, **__: 0.0),
    })

    # Containers
    ppo_caps, ppo_vars, ppo_sw = [], [], []
    dqn_caps, dqn_vars, dqn_sw = [], [], []
    hyb_caps, hyb_vars, hyb_sw = [], [], []
    hyb_ppo_share, hyb_dqn_share, hyb_budget_eff = [], [], []

    for e in range(episodes):
        s = seed + e

        # PPO baseline
        env = BatteryPackEnv(seed=s, profile=profile)
        r_ppo = run_controller(
            env,
            policy_fn=lambda env_, obs_: (act_ppo(ppo_model, obs_), "PPO"),
            seed=s
        )
        ppo_caps.append(r_ppo["capacity_mAh"])
        ppo_vars.append(r_ppo["soc_variance"])
        ppo_sw.append(r_ppo["switches"])

        # DQN baseline
        env = BatteryPackEnv(seed=s, profile=profile)
        r_dqn = run_controller(
            env,
            policy_fn=lambda env_, obs_: (act_dqn(dqn_model, obs_), "DQN"),
            seed=s
        )
        dqn_caps.append(r_dqn["capacity_mAh"])
        dqn_vars.append(r_dqn["soc_variance"])
        dqn_sw.append(r_dqn["switches"])

        # Hybrid-V2
        env = BatteryPackEnv(seed=s, profile=profile)
        hyb = AdaptiveHybrid(
            ppo_path=ppo_path, dqn_path=dqn_path,
            k=k, theta=theta, max_switch_budget=max_switch_budget,
            decay=decay, window=window,
            min_dqn_span=min_dqn_span, min_ppo_span=min_ppo_span,
            seed=s
        )
        r_hyb = run_hybrid(env, hyb, seed=s)
        hyb_caps.append(r_hyb["capacity_mAh"])
        hyb_vars.append(r_hyb["soc_variance"])
        hyb_sw.append(r_hyb["switches"])
        hyb_ppo_share.append(r_hyb["ppo_step_share"])
        hyb_dqn_share.append(r_hyb["dqn_step_share"])
        hyb_budget_eff.append(r_hyb["allowed_switch_budget_effective"])

    def stats(arr):
        a = np.array(arr, dtype=float)
        return float(np.mean(a)), float(np.std(a))

    out["baselines"]["PPO"] = {
        "capacity_mAh_mean": stats(ppo_caps)[0],
        "capacity_mAh_std":  stats(ppo_caps)[1],
        "soc_variance_mean": stats(ppo_vars)[0],
        "soc_variance_std":  stats(ppo_vars)[1],
        "switches_mean":     stats(ppo_sw)[0],
        "switches_std":      stats(ppo_sw)[1],
    }
    out["baselines"]["DQN"] = {
        "capacity_mAh_mean": stats(dqn_caps)[0],
        "capacity_mAh_std":  stats(dqn_caps)[1],
        "soc_variance_mean": stats(dqn_vars)[0],
        "soc_variance_std":  stats(dqn_vars)[1],
        "switches_mean":     stats(dqn_sw)[0],
        "switches_std":      stats(dqn_sw)[1],
    }
    out["hybrid"] = {
        "capacity_mAh_mean": stats(hyb_caps)[0],
        "capacity_mAh_std":  stats(hyb_caps)[1],
        "soc_variance_mean": stats(hyb_vars)[0],
        "soc_variance_std":  stats(hyb_vars)[1],
        "switches_mean":     stats(hyb_sw)[0],
        "switches_std":      stats(hyb_sw)[1],
        "ppo_step_share_mean": stats(hyb_ppo_share)[0],
        "dqn_step_share_mean": stats(hyb_dqn_share)[0],
        "budget_effective_mean": stats(hyb_budget_eff)[0],
        # keep episode arrays too in case we need CIs later
        "episodes": {
            "capacity_mAh": hyb_caps,
            "soc_variance": hyb_vars,
            "switches": hyb_sw,
            "ppo_step_share": hyb_ppo_share,
            "dqn_step_share": hyb_dqn_share
        }
    }

    return out


# ---------------------------- Plot & Save ----------------------------

def save_outputs(summary: Dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # JSON
    with open(os.path.join(outdir, "hybrid_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV (short)
    import csv
    hdr = ["Controller", "Capacity_mAh_mean", "Capacity_mAh_std",
           "SoC_variance_mean", "SoC_variance_std", "Switches_mean", "Switches_std"]
    rows = []
    for name in ["PPO", "DQN"]:
        b = summary["baselines"][name]
        rows.append([name, b["capacity_mAh_mean"], b["capacity_mAh_std"],
                     b["soc_variance_mean"], b["soc_variance_std"],
                     b["switches_mean"], b["switches_std"]])
    h = summary["hybrid"]
    rows.append(["HybridV2", h["capacity_mAh_mean"], h["capacity_mAh_std"],
                 h["soc_variance_mean"], h["soc_variance_std"],
                 h["switches_mean"], h["switches_std"]])

    with open(os.path.join(outdir, "hybrid_v2_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)

    # Chart
    # Bars: capacity; Line (right axis): SoC std; Markers: switches
    labels = ["PPO", "DQN", "HybridV2"]
    cap = [summary["baselines"]["PPO"]["capacity_mAh_mean"],
           summary["baselines"]["DQN"]["capacity_mAh_mean"],
           summary["hybrid"]["capacity_mAh_mean"]]
    var = [summary["baselines"]["PPO"]["soc_variance_mean"],
           summary["baselines"]["DQN"]["soc_variance_mean"],
           summary["hybrid"]["soc_variance_mean"]]
    std = [math.sqrt(max(v, 0.0)) for v in var]
    sw  = [summary["baselines"]["PPO"]["switches_mean"],
           summary["baselines"]["DQN"]["switches_mean"],
           summary["hybrid"]["switches_mean"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(9,5))
    ax1.bar(x, cap, width, label="Usable Capacity (mAh)", color="tab:blue")
    ax1.set_ylabel("Usable Capacity (mAh)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Controller")

    ax2 = ax1.twinx()
    ax2.plot(x, std, marker="o", linewidth=2.5, color="tab:red", label="SoC Std. Dev")
    ax2.set_ylabel("SoC Std. Dev", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Switch markers as text
    for xi, swi in zip(x, sw):
        ax1.text(xi, cap[xi] + 0.02 * max(cap), f"sw={swi:.1f}", ha="center", va="bottom", fontsize=9, rotation=0)

    plt.title("PPO vs DQN vs Hybrid-V2 (Adaptive Arbiter)")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "hybrid_v2_comparison.png"), dpi=300)
    plt.close(fig)


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", required=True, type=str)
    ap.add_argument("--dqn-ckpt", required=True, type=str)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--profile", type=str, default="charge-rest-discharge")

    # Adaptive gating knobs
    ap.add_argument("--k", type=float, default=40.0, help="Softmax slope for DQN probability")
    ap.add_argument("--theta", type=float, default=0.0045, help="Soft trigger (on urgency) for DQN")
    ap.add_argument("--max-switch-budget", type=int, default=60, help="Hard per-episode switch cap")
    ap.add_argument("--decay", type=float, default=0.98, help="Budget decay when variance remains low")
    ap.add_argument("--window", type=int, default=15, help="Running variance window (steps)")
    ap.add_argument("--min-dqn-span", type=int, default=6, help="Min steps to stick with DQN once chosen")
    ap.add_argument("--min-ppo-span", type=int, default=6, help="Min steps to stick with PPO once chosen")
    args = ap.parse_args()

    outdir = "results_hybrid_v2"
    os.makedirs(outdir, exist_ok=True)

    summary = eval_all(
        ppo_path=args.ppo_ckpt,
        dqn_path=args.dqn_ckpt,
        episodes=args.episodes,
        seed=args.seed,
        profile=args.profile,
        k=args.k, theta=args.theta,
        max_switch_budget=args.max_switch_budget,
        decay=args.decay,
        window=args.window,
        min_dqn_span=args.min_dqn_span,
        min_ppo_span=args.min_ppo_span
    )

    save_outputs(summary, outdir)
    print("\n[Hybrid-V2] Done.")
    print(f"- JSON: {os.path.join(outdir, 'hybrid_v2_summary.json')}")
    print(f"- CSV : {os.path.join(outdir, 'hybrid_v2_summary.csv')}")
    print(f"- PNG : {os.path.join(outdir, 'hybrid_v2_comparison.png')}\n")


if __name__ == "__main__":
    main()
