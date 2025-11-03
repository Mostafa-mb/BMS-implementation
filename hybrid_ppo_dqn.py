import argparse, os, json, math
import numpy as np
from stable_baselines3 import PPO, DQN
from gym_bms import BatteryPackEnv

# ---------- small helpers ----------
def count_switch_changes(prev_bits, bits):
    return int(np.sum(prev_bits != bits))

def variance_excluding_nans(arr):
    a = np.array(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.var(a)) if a.size else 0.0

# ---------- Hybrid policy ----------
class HybridEnsemble:
    """
    Arbiter-Gated Ensemble (AGE): prefers PPO, escalates to DQN only when needed.
    """
    def __init__(
        self,
        ppo_path: str,
        dqn_path: str,
        var_hi: float = 0.005,   # switch to DQN if var >= var_hi
        var_lo: float = 0.003,   # switch back to PPO if var <= var_lo (hysteresis)
        switch_budget: int = 40, # max allowed switch flips per episode
        cooldown_steps: int = 8  # stick with selected expert for N steps
    ):
        self.ppo = PPO.load(ppo_path, custom_objects={
            "lr_schedule": (lambda *_, **__: 0.0),
            "clip_range":  (lambda *_, **__: 0.2),
        })
        self.dqn = DQN.load(dqn_path, custom_objects={
            "lr_schedule":          (lambda *_, **__: 0.0),
            "exploration_schedule": (lambda *_, **__: 0.0),
        })
        self.var_hi = var_hi
        self.var_lo = var_lo
        self.switch_budget = switch_budget
        self.cooldown_steps = cooldown_steps

        self._expert = "PPO"
        self._cooldown_left = 0
        self._switches_used = 0
        self._last_bits = None

    def reset_episode(self):
        self._expert = "PPO"
        self._cooldown_left = 0
        self._switches_used = 0
        self._last_bits = None

    def _choose_expert(self, var_soc, budget_left):
        # Hysteresis window on variance:
        if self._expert == "PPO":
            if var_soc >= self.var_hi and budget_left > 0 and self._cooldown_left == 0:
                self._expert = "DQN"
                self._cooldown_left = self.cooldown_steps
        else:  # currently DQN
            if var_soc <= self.var_lo and self._cooldown_left == 0:
                self._expert = "PPO"
                self._cooldown_left = self.cooldown_steps

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # If budget is exhausted, force PPO (calmer)
        if budget_left <= 0:
            self._expert = "PPO"

        return self._expert

    def act(self, obs, env):
        # compute current SoC variance from env
        var_soc = variance_excluding_nans(env.soc)

        # compute remaining budget based on actual flips observed
        if self._last_bits is None:
            budget_left = self.switch_budget
        else:
            budget_left = max(0, self.switch_budget - self._switches_used)

        expert = self._choose_expert(var_soc, budget_left)

        if expert == "PPO":
            a, _ = self.ppo.predict(obs, deterministic=True)
        else:
            a, _ = self.dqn.predict(obs, deterministic=True)

        # we cannot update _switches_used until after env.step() returns new bits
        return int(a), expert

    def update_switch_count(self, new_bits):
        if self._last_bits is None:
            self._last_bits = new_bits.copy()
            return
        self._switches_used += count_switch_changes(self._last_bits, new_bits)
        self._last_bits = new_bits.copy()

# ---------- evaluation ----------
def run_episode(env, policy: HybridEnsemble, seed=123):
    obs, _ = env.reset(seed=seed)
    policy.reset_episode()

    dt_h = env.dt / 3600.0
    Q_Ah = 0.0
    expert_steps = {"PPO": 0, "DQN": 0}

    done = False
    info_last = {}
    while not done:
        a, expert = policy.act(obs, env)
        obs, reward, done, trunc, info = env.step(a)

        # capacity: integrate absolute current (robust across sign conventions)
        I = 0.0
        if isinstance(info, dict) and "I_load" in info:
            I = abs(float(info["I_load"]))
        elif hasattr(env, "I_load"):
            I = abs(float(getattr(env, "I_load")))
        elif hasattr(env, "_profile_current"):
            I = abs(float(env._profile_current(env.t)))

        Q_Ah += I * dt_h
        expert_steps[expert] += 1

        # update switch budget usage
        policy.update_switch_count(env.switch_on)
        info_last = info

    # final metrics
    var_soc_end = variance_excluding_nans(env.soc)
    return {
        "usable_capacity_mAh": Q_Ah * 1000.0,
        "soc_variance": var_soc_end,
        "switches_used": policy._switches_used,
        "expert_steps": expert_steps
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", required=True, type=str)
    ap.add_argument("--dqn-ckpt", required=True, type=str)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--profile", type=str, default="charge-rest-discharge")
    # arbiter knobs:
    ap.add_argument("--var-hi", type=float, default=0.005)
    ap.add_argument("--var-lo", type=float, default=0.003)
    ap.add_argument("--switch-budget", type=int, default=40)
    ap.add_argument("--cooldown", type=int, default=8)
    args = ap.parse_args()

    # init
    os.makedirs("results_hybrid", exist_ok=True)
    arb = HybridEnsemble(
        ppo_path=args.ppo_ckpt,
        dqn_path=args.dqn_ckpt,
        var_hi=args.var_hi, var_lo=args.var_lo,
        switch_budget=args.switch_budget,
        cooldown_steps=args.cooldown
    )

    # eval
    caps, vars_, flips, ppo_steps, dqn_steps = [], [], [], [], []
    for k in range(args.episodes):
        env = BatteryPackEnv(seed=args.seed + k, profile=args.profile)
        metrics = run_episode(env, arb, seed=args.seed + k)
        caps.append(metrics["usable_capacity_mAh"])
        vars_.append(metrics["soc_variance"])
        flips.append(metrics["switches_used"])
        ppo_steps.append(metrics["expert_steps"]["PPO"])
        dqn_steps.append(metrics["expert_steps"]["DQN"])

    summary = {
        "params": {
            "var_hi": args.var_hi, "var_lo": args.var_lo,
            "switch_budget": args.switch_budget, "cooldown": args.cooldown
        },
        "results": {
            "capacity_mean_mAh": float(np.mean(caps)),
            "capacity_std_mAh": float(np.std(caps)),
            "soc_variance_mean": float(np.mean(vars_)),
            "soc_variance_std": float(np.std(vars_)),
            "switches_mean": float(np.mean(flips)),
            "switches_std": float(np.std(flips)),
            "ppo_step_share_mean": float(np.mean(np.array(ppo_steps) / (np.array(ppo_steps) + np.array(dqn_steps) + 1e-9))),
            "dqn_step_share_mean": float(np.mean(np.array(dqn_steps) / (np.array(ppo_steps) + np.array(dqn_steps) + 1e-9))),
        }
    }

    with open("results_hybrid/hybrid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[Hybrid summary]")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
