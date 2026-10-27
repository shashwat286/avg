import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=20):
    if len(data) < window_size:
        window_size = max(1, len(data) // 5)
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    continual_dir = "./results_continual"
    live_csv = os.path.join(continual_dir, "diagnostics_live.csv")

    if not os.path.exists(live_csv):
        csv_files = glob.glob(f"{continual_dir}/*_diagnostics.csv")
        if csv_files:
            live_csv = max(csv_files, key=os.path.getmtime)
        else:
            print("No diagnostics CSV found yet in results_continual/. Training may be starting up.")
            return

    rows = []
    try:
        with open(live_csv, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: float(v) if k not in ("stage", "task") else v for k, v in r.items()})
    except Exception as e:
        print(f"Error reading CSV {live_csv}: {e}")
        return

    if len(rows) < 2:
        print(f"Diagnostics CSV currently has {len(rows)} entries. Waiting for more data...")
        return

    print("==========================================================================================")
    print("                     CONTINUAL AVG: ACTOR & CRITIC MONITOR                                 ")
    print("==========================================================================================")
    print(f"Loaded: {live_csv} ({len(rows):,} logged episodes)")
    
    latest = rows[-1]
    print(f"Current State: [{latest['stage']}] Task: {latest['task']} | Episode {int(latest['episode'])}, Step {int(latest['step']):,}")
    print(f"Episode Return: {latest['return']:.2f} (Length: {int(latest['length'])}) | Scaled TD Error (sigma): {latest['sigma']:.3f}")
    print("------------------------------------------------------------------------------------------")
    print("GRADIENTS (L2 Norm):")
    print(f"  Actor  -> Core: {latest['act_core_grad']:.4f} | Current Task Head: {latest['act_head_grad']:.4f}")
    print(f"  Critic -> Core: {latest['crit_core_grad']:.4f} | Current Task Head: {latest['crit_head_grad']:.4f}")
    print("PARAMETERS (L2 Norm):")
    print(f"  Actor  -> Core: {latest['act_core_param']:.3f} | Current Task Head: {latest['act_head_param']:.3f}")
    print(f"  Critic -> Core: {latest['crit_core_param']:.3f} | Current Task Head: {latest['crit_head_param']:.3f}")
    if latest['act_core_drift'] > 0 or latest['crit_core_drift'] > 0:
        print("CORE DRIFT (Distance from Humanoid Stage 1 Weights):")
        print(f"  Actor Core Drift : {latest['act_core_drift']:.4f}")
        print(f"  Critic Core Drift: {latest['crit_core_drift']:.4f}")
    print("------------------------------------------------------------------------------------------")

    # Arrays for plotting
    steps = np.array([r['step'] for r in rows])
    act_core_grad = np.array([r['act_core_grad'] for r in rows])
    act_head_grad = np.array([r['act_head_grad'] for r in rows])
    crit_core_grad = np.array([r['crit_core_grad'] for r in rows])
    crit_head_grad = np.array([r['crit_head_grad'] for r in rows])
    act_core_param = np.array([r['act_core_param'] for r in rows])
    crit_core_param = np.array([r['crit_core_param'] for r in rows])
    act_head_param = np.array([r['act_head_param'] for r in rows])
    crit_head_param = np.array([r['crit_head_param'] for r in rows])
    act_core_drift = np.array([r['act_core_drift'] for r in rows])
    crit_core_drift = np.array([r['crit_core_drift'] for r in rows])
    ploss = np.array([r['ploss'] for r in rows])
    qloss = np.array([r['qloss'] for r in rows])
    sigma = np.array([r['sigma'] for r in rows])
    returns = np.array([r['return'] for r in rows])

    # 6-Panel Visualization Dashboard
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    w = max(1, min(30, len(rows) // 10))

    # Panel 1: Actor Gradient Norm (Core vs Head)
    ax1 = axes[0, 0]
    ax1.plot(steps, act_core_grad, color="#1f77b4", alpha=0.3, label="Actor Core Grad (raw)")
    ax1.plot(steps, act_head_grad, color="#ff7f0e", alpha=0.3, label="Actor Head Grad (raw)")
    if len(rows) >= w:
        ax1.plot(steps[w-1:], moving_average(act_core_grad, w), color="#08519c", linewidth=2, label="Actor Core Grad (smooth)")
        ax1.plot(steps[w-1:], moving_average(act_head_grad, w), color="#bd0026", linewidth=2, label="Actor Head Grad (smooth)")
    ax1.set_title("Actor Gradient Norm: Shared Core vs Task Head", fontweight="bold")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("||∇θ||_2")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Critic Gradient Norm (Core vs Head)
    ax2 = axes[0, 1]
    ax2.plot(steps, crit_core_grad, color="#2ca02c", alpha=0.3, label="Critic Core Grad (raw)")
    ax2.plot(steps, crit_head_grad, color="#9467bd", alpha=0.3, label="Critic Head Grad (raw)")
    if len(rows) >= w:
        ax2.plot(steps[w-1:], moving_average(crit_core_grad, w), color="#006d2c", linewidth=2, label="Critic Core Grad (smooth)")
        ax2.plot(steps[w-1:], moving_average(crit_head_grad, w), color="#54278f", linewidth=2, label="Critic Head Grad (smooth)")
    ax2.set_title("Critic Gradient Norm: Shared Core vs Task Head", fontweight="bold")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("||∇θ||_2")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Core Representation Drift
    ax3 = axes[1, 0]
    has_drift = (act_core_drift > 0).any()
    if has_drift:
        ax3.plot(steps, act_core_drift, color="#d95f02", linewidth=2, label="Actor Core Drift ||θ - θ_Stage1||")
        ax3.plot(steps, crit_core_drift, color="#7570b3", linewidth=2, label="Critic Core Drift ||θ - θ_Stage1||")
        ax3.legend(loc="upper left", frameon=True)
    else:
        ax3.text(0.5, 0.5, "Stage 1 Active:\nBaseline Core Reference Being Formed\n(Drift will plot during Stage 2)", ha="center", va="center", color="gray", fontsize=11)
    ax3.set_title("Shared Core Representation Drift from Task 1", fontweight="bold")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("||θ(t) - θ_Stage1||_2")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Parameter Norms
    ax4 = axes[1, 1]
    ax4.plot(steps, act_core_param, color="#08519c", label="Actor Core Norm")
    ax4.plot(steps, crit_core_param, color="#006d2c", label="Critic Core Norm")
    ax4.plot(steps, act_head_param, color="#fc4e2a", linestyle="--", label="Actor Head Norm")
    ax4.plot(steps, crit_head_param, color="#88419d", linestyle="--", label="Critic Head Norm")
    ax4.set_title("Parameter L2 Norms (Core & Heads)", fontweight="bold")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("||θ||_2")
    ax4.legend(loc="upper left", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    # Panel 5: Losses & TD Error Scaling
    ax5 = axes[2, 0]
    if len(rows) >= w:
        ax5.plot(steps[w-1:], moving_average(ploss, w), color="#e7298a", label="Policy Loss (AVG)")
        ax5.plot(steps[w-1:], moving_average(qloss, w), color="#1f78b4", label="Q Loss (Scaled)")
    ax5.plot(steps, sigma, color="#33a02c", linestyle=":", label="TD Scaler σ")
    ax5.set_title("Losses & TD Error Scaler σ", fontweight="bold")
    ax5.set_xlabel("Steps")
    ax5.set_ylabel("Value")
    ax5.legend(loc="upper right", frameon=True)
    ax5.grid(True, linestyle="--", alpha=0.6)

    # Panel 6: Episode Returns
    ax6 = axes[2, 1]
    ax6.plot(steps, returns, color="teal", alpha=0.3, label="Return (raw)")
    if len(rows) >= w:
        ax6.plot(steps[w-1:], moving_average(returns, w), color="darkcyan", linewidth=2, label="Return (smooth)")
    ax6.set_title("Episode Return", fontweight="bold")
    ax6.set_xlabel("Steps")
    ax6.set_ylabel("Return")
    ax6.legend(loc="upper left", frameon=True)
    ax6.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_img = os.path.join(continual_dir, "diagnostics_dashboard.png")
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"Updated live dashboard saved to: {out_img}")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()
