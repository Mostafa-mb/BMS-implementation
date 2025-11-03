import os, json, csv, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
# ----------------------------------

def load_metrics():
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("summary.csv not found. Run scripts.report first.")
    df = pd.read_csv(csv_path)
    return df

def plot_soc_switch_pairs():
    """Combine SoC & switching for both algos & profiles, similar to Figs. 11–14."""
    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    combos = [
        ("dqn_discharge-rest-charge", "DQN — discharge–rest–charge"),
        ("dqn_charge-rest-discharge", "DQN — charge–rest–discharge"),
        ("ppo_discharge-rest-charge", "PPO — discharge–rest–charge"),
        ("ppo_charge-rest-discharge", "PPO — charge–rest–discharge"),
    ]
    for ax, (prefix, title) in zip(axs.flatten(), combos):
        soc_path = f"{RESULTS_DIR}/{prefix}_soc.npy"
        sw_path = f"{RESULTS_DIR}/{prefix}_switch.npy"
        if not os.path.exists(soc_path):
            continue
        soc = np.load(soc_path)
        sw = np.load(sw_path)
        t = np.arange(soc.shape[0]) * 30.0
        ax2 = ax.twinx()
        for i in range(5):
            ax.plot(t, soc[:,i]*100, label=f"SoC Cell {i+1}")
            ax2.step(t, sw[:,i], where='post', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SoC (%)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/figure11_14_soc_switch.png", dpi=250)
    plt.close(fig)
    print("[✓] Saved combined SoC+Switch plot (Fig. 11–14 style)")

def make_metrics_table(df):
    """Generate publication-style table."""
    out_csv = os.path.join(RESULTS_DIR, "table_metrics.csv")
    out_tex = os.path.join(RESULTS_DIR, "table_metrics_latex.txt")

    df_fmt = df.copy()
    df_fmt["Var(SoC)"] = df_fmt["Var(SoC)_mean"].round(6)
    df_fmt["Switches"] = df_fmt["Switches_mean"].round(2)
    df_fmt["Working Ratio"] = df_fmt["Working_ratio_mean"].round(3)
    df_fmt = df_fmt[["Algorithm","Profile","Var(SoC)","Switches","Working Ratio"]]
    df_fmt.to_csv(out_csv, index=False)

    # Make LaTeX
    tex = "\\begin{table}[h!]\n\\centering\n"
    tex += "\\caption{Comparison of DQN and PPO performance on SoC balancing under two profiles}\n"
    tex += "\\begin{tabular}{lccc}\n\\toprule\n"
    tex += "Algorithm & Profile & Var(SoC) & Switches \\\\\n\\midrule\n"
    for _,r in df_fmt.iterrows():
        tex += f"{r['Algorithm']} & {r['Profile'].replace('-','--')} & {r['Var(SoC)']:.4f} & {r['Switches']:.1f} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(out_tex, "w") as f:
        f.write(tex)

    print("[✓] Saved table_metrics.csv and LaTeX-ready table")

if __name__ == "__main__":
    df = load_metrics()
    make_metrics_table(df)
    plot_soc_switch_pairs()
    print("[✓] Publication-style report completed.")
