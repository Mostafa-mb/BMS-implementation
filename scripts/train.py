import os, argparse, time, glob, re
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from gym_bms import BatteryPackEnv

def make_env(seed, profile="discharge-rest-charge"):
    env = BatteryPackEnv(seed=seed, profile=profile)
    return Monitor(env)

def set_seeds(seed):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _latest_checkpoint(path):
    """Return path to latest checkpoint in `path` or None."""
    if not os.path.isdir(path):
        return None
    # Prefer rl_model_*_steps.zip with highest steps
    ckpts = glob.glob(os.path.join(path, "rl_model_*_steps.zip"))
    def steps_of(p):
        m = re.search(r"rl_model_(\d+)_steps\.zip$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpts = sorted(ckpts, key=steps_of, reverse=True)
    if ckpts:
        return ckpts[0]
    # Fallbacks
    for name in ["best_model.zip", "final_model.zip"]:
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn","ppo"], required=True)
    parser.add_argument("--total-steps", type=int, default=8_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1_000_000, help="steps between eval + checkpoint")
    parser.add_argument("--profile", choices=["discharge-rest-charge","charge-rest-discharge"], default="discharge-rest-charge")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint if available")
    args = parser.parse_args()

    set_seeds(args.seed)
    env = make_env(args.seed, profile=args.profile)

    # Logging
    run_name = f"{args.algo}_seed{args.seed}_{int(time.time())}"
    logdir = os.path.join("runs", run_name)
    os.makedirs(logdir, exist_ok=True)
    new_logger = configure(logdir, ["stdout","tensorboard"])

    # Checkpoints
    ckpt_dir = os.path.join("checkpoints", f"{args.algo}_seed{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=args.eval_interval, save_path=ckpt_dir, name_prefix="rl_model")

    # Eval callback on the alt profile
    eval_env = make_env(args.seed+1, profile="charge-rest-discharge")
    eval_cb = EvalCallback(eval_env, best_model_save_path=ckpt_dir, eval_freq=args.eval_interval, n_eval_episodes=5, deterministic=True, render=False)

    # Create or resume model
    model = None
    resume_path = _latest_checkpoint(ckpt_dir) if args.resume else None

    if args.algo == "dqn":
        if resume_path:
            model = DQN.load(resume_path, env=env)
        else:
            model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=200000,
                        batch_size=256, train_freq=(4, "step"), target_update_interval=10_000,
                        exploration_initial_eps=0.1, exploration_final_eps=0.05, exploration_fraction=0.1,
                        tensorboard_log=logdir, seed=args.seed)
    else:
        if resume_path:
            model = PPO.load(resume_path, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256, learning_rate=3e-4,
                        gae_lambda=0.95, gamma=0.995, clip_range=0.2, ent_coef=0.0,
                        tensorboard_log=logdir, seed=args.seed)

    model.set_logger(new_logger)

    # If resuming, continue step count (no reset)
    reset_flag = not bool(resume_path)

    model.learn(total_timesteps=args.total_steps,
                log_interval=args.log_interval,
                callback=[checkpoint_callback, eval_cb],
                reset_num_timesteps=reset_flag)

    model.save(os.path.join(ckpt_dir, "final_model"))

if __name__ == "__main__":
    main()
