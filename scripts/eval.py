import os, argparse, json
import numpy as np
from stable_baselines3 import DQN, PPO
from gym_bms import BatteryPackEnv

def run_eval(model, profile, seed, episodes=10):
    env = BatteryPackEnv(seed=seed, profile=profile)
    metrics = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed+ep)
        done = False
        ep_sw = 0
        soc_trace = []
        sw_trace = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            ep_sw += info.get("switches", 0)
            soc_trace.append(env.soc.copy())
            sw_trace.append(env.switch_on.copy())
        var_soc = float(np.var(env.soc))
        metrics.append({"var_soc": var_soc, "switch_changes": ep_sw})
        # Save traces
        np.save(f"results/{profile}_seed{seed}_ep{ep}_soc.npy", np.array(soc_trace))
        np.save(f"results/{profile}_seed{seed}_ep{ep}_switch.npy", np.array(sw_trace))
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn","ppo"], required=True)
    parser.add_argument("--checkpoint", required=True, help="path to best_model.zip or final_model.zip")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.algo == "dqn":
        model = DQN.load(args.checkpoint)
    else:
        model = PPO.load(args.checkpoint)

    all_metrics = {}
    for profile in ["discharge-rest-charge", "charge-rest-discharge"]:
        m = run_eval(model, profile, seed=args.seed, episodes=10)
        all_metrics[profile] = m

    with open("results/metrics_{}.json".format(args.algo), "w") as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()
