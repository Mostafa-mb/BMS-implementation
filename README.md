# BMS-RL Reproduction Scaffold (DQN vs PPO)
This repo gives you *plug-and-play* code to recreate the experiments from the paper you shared on **RL for passive balancing** with 5 Li-ion cells in series, 6Ω shunt resistors, 30 s decision interval, and the multi-objective reward. It logs to **TensorBoard**, saves **checkpoints**, fixes **seeds**, and exports publication-style **plots**.

> ⚠️ **About exact replication**: The paper draws ECM parameters (R0,R1,C1,R2,C2) and the OCV–SoC curve from Tran et al. (2021). We include the (R,C) table from the appendix and a **placeholder OCV** curve (`data/ocv_table.csv`). To match numbers and plots precisely, replace that CSV with the OCV table from Tran et al. If you don't, you’ll still get consistent trends and apples-to-apples DQN vs PPO comparisons—but absolute values may differ.

## Quick start

1) (Optional but recommended) Create a Python 3.10+ venv.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) **Train** DQN and PPO (logs -> `runs/*` for TensorBoard, checkpoints -> `checkpoints/*`).

```bash
# DQN
python scripts/train.py --algo dqn --total-steps 8000000 --seed 42

# PPO
python scripts/train.py --algo ppo --total-steps 15000000 --seed 42
```

3) **TensorBoard** (training curves like Fig. 9):  
```bash
tensorboard --logdir runs
```

4) **Evaluate** (both load profiles; exports CSV metrics and SoC/voltage/current/switch traces):  
```bash
python scripts/eval.py --algo ppo --checkpoint checkpoints/ppo_seed42/best_model.zip
python scripts/eval.py --algo dqn --checkpoint checkpoints/dqn_seed42/best_model.zip
```

5) **Plot** publication-style figures (SoC, switching, training curves):  
```bash
python scripts/plot_results.py
```

All outputs go to `plots/` and `results/` folders.

## Replacing the OCV table (to match the paper exactly)
Edit `data/ocv_table.csv` to the OCV vs SoC mapping from Tran et al. (2021) for the NCR18650B/NCA cell. The first column must be **SoC in [0,1]** and the second column the **OCV in volts**. We interpolate internally.

## Repro knobs
- `--seed` controls seeds for `numpy`, `random`, `torch`, and env.
- `--total-steps` training steps.
- `--log-interval` and `--eval-interval` for checkpointing & TensorBoard.
- `--profile` selects the load profile: `discharge-rest-charge` (training default) or `charge-rest-discharge` (test profile).

## Files
- `gym_bms/env.py` — Gymnasium environment (ECM, state, action, reward, retracing, normalization).
- `gym_bms/params.py` — ECM tables (R0,R1,C1,R2,C2) from the paper appendix + linear interpolation.
- `data/ocv_table.csv` — Placeholder OCV–SoC table. Replace for exact replication.
- `scripts/train.py` — Train DQN/PPO with Stable-Baselines3; TensorBoard + checkpoints.
- `scripts/eval.py` — Deterministic evaluation on both profiles; exports metrics.
- `scripts/plot_results.py` — Plots matching key paper figures (SoC traces, switching, training curves).

## License
For your thesis/research. Cite the original paper accordingly.
