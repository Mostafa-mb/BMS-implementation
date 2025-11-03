
import os, argparse, csv
import numpy as np
from gym_bms import BatteryPackEnv
from stable_baselines3 import PPO, DQN

def usable_capacity_mAh(model, profile, seed=123):
    env = BatteryPackEnv(seed=seed, profile=profile)
    obs, _ = env.reset(seed=seed)
    done = False
    Q_Ah = 0.0
    dt_h = env.dt / 3600.0  # seconds -> hours
    while not done:
        if model is None:
            action = 0  # baseline: all switches OFF (no balancing)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        I = info.get("I_load", 0.0)
        if I > 0:            # discharge only
            Q_Ah += I * dt_h # Ah increment for this step
    return Q_Ah * 1000.0     # mAh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","dqn"], help="Model algorithm")
    parser.add_argument("--ckpt", type=str, help="Path to trained checkpoint")
    parser.add_argument("--baseline", action="store_true", help="Compute no-balancing baseline")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load model or use baseline
    if args.baseline:
        model = None
        label = "NoBalancing"
    else:
        if args.algo == "ppo":
            model = PPO.load(args.ckpt)
            label = "PPO"
        elif args.algo == "dqn":
            model = DQN.load(args.ckpt)
            label = "DQN"
        else:
            raise ValueError("Specify --baseline or (--algo and --ckpt)")

    rows = []
    for profile in ["discharge-rest-charge", "charge-rest-discharge"]:
        cap = usable_capacity_mAh(model, profile, seed=args.seed)
        rows.append([label, profile, round(cap, 2)])

    out_csv = "results/usable_capacity.csv"
    header = ["Algorithm","Profile","UsableCapacity_mAh"]
    if os.path.exists(out_csv):
        # append or overwrite label entries
        # simple approach: rewrite filtering out same label
        old = []
        with open(out_csv, newline="") as f:
            rd = csv.reader(f)
            hdr = next(rd, None)
            for r in rd:
                if r and r[0] != label:
                    old.append(r)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for r in old + rows: w.writerow(r)
    else:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    print(f"[✓] Wrote {out_csv} with {label} rows.")

if __name__ == "__main__":
    main()
