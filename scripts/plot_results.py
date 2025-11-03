import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

def plot_soc_traces(pattern, title, out_png):
    files = sorted(glob.glob(pattern))[:1]  # plot first episode file for clarity
    if not files:
        return
    arr = np.load(files[0])  # T x 5
    T = arr.shape[0]
    t = np.arange(T) * 30.0
    for i in range(5):
        plt.plot(t, arr[:, i]*100.0, label=f"Cell {i+1}")
    plt.xlabel("Time (s)"); plt.ylabel("SoC (%)")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_switch_traces(pattern, title, out_png):
    files = sorted(glob.glob(pattern))[:1]
    if not files:
        return
    arr = np.load(files[0])  # T x 5 (0/1)
    T = arr.shape[0]
    t = np.arange(T) * 30.0
    for i in range(5):
        plt.step(t, arr[:, i], where='post', label=f"Cell {i+1}")
    plt.xlabel("Time (s)"); plt.ylabel("Switch (0/1)")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def main():
    # Example: plot SoC and switch behavior (similar to Figs. 11–14)
    plot_soc_traces("results/discharge-rest-charge_seed*_ep0_soc.npy",
                    "SoC over time (Discharge–Rest–Charge)", "plots/soc_drc.png")
    plot_soc_traces("results/charge-rest-discharge_seed*_ep0_soc.npy",
                    "SoC over time (Charge–Rest–Discharge)", "plots/soc_crd.png")
    plot_switch_traces("results/discharge-rest-charge_seed*_ep0_switch.npy",
                       "Switching (Discharge–Rest–Charge)", "plots/switch_drc.png")
    plot_switch_traces("results/charge-rest-discharge_seed*_ep0_switch.npy",
                       "Switching (Charge–Rest–Discharge)", "plots/switch_crd.png")

if __name__ == "__main__":
    main()
