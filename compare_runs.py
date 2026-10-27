import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=50):
    if len(data) < window_size:
        window_size = max(1, len(data) // 5)
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    baseline_dir = "./results_baseline"
    continual_dir = "./results_continual"

    print("==================================================================")
    print("      Action Value Gradient (AVG): Baseline vs Continual Comparison")
    print("==================================================================")

    # 1. Load Baseline Data
    baseline_pkls = glob.glob(f"{baseline_dir}/*_seed-*.pkl")
    baseline_data = None
    if baseline_pkls:
        latest_baseline = max(baseline_pkls, key=os.path.getmtime)
        try:
            with open(latest_baseline, "rb") as f:
                baseline_data = pickle.load(f)
            print(f"[Baseline] Loaded: {latest_baseline}")
        except Exception as e:
            print(f"[Baseline] Warning: Failed reading {latest_baseline} ({e})")
    else:
        print("[Baseline] No completed summary pkl found yet in results_baseline/.")

    # 2. Load Continual Data
    continual_pkls = glob.glob(f"{continual_dir}/*_summary.pkl")
    continual_data = None
    if continual_pkls:
        latest_continual = max(continual_pkls, key=os.path.getmtime)
        try:
            with open(latest_continual, "rb") as f:
                continual_data = pickle.load(f)
            print(f"[Continual] Loaded: {latest_continual}")
        except Exception as e:
            print(f"[Continual] Warning: Failed reading {latest_continual} ({e})")
    else:
        print("[Continual] No completed summary pkl found yet in results_continual/.")

    # 3. Create Comparison Plots if data is available
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Humanoid Performance (Baseline Monolithic vs Continual Head 1)
    ax1 = axes[0]
    ax1.set_title("Task 1 (Humanoid-v4): Learning Curve Comparison", fontweight="bold", fontsize=11)
    has_b = False
    has_c = False

    if baseline_data is not None:
        b_steps, b_rets, b_env = baseline_data
        if len(b_rets) > 0:
            b_cum_steps = np.cumsum(b_steps)
            ax1.plot(b_cum_steps, b_rets, color="gray", alpha=0.25, label="Baseline Humanoid (raw)")
            w_b = max(1, min(50, len(b_rets) // 10))
            if len(b_rets) >= w_b:
                ax1.plot(b_cum_steps[w_b-1:], moving_average(b_rets, w_b), color="navy", linewidth=2, label="Baseline Humanoid (10M)")
            has_b = True

    if continual_data is not None:
        c_steps = continual_data["t1_steps"]
        c_rets = continual_data["t1_returns"]
        if len(c_rets) > 0:
            c_cum_steps = np.cumsum(c_steps)
            ax1.plot(c_cum_steps, c_rets, color="teal", alpha=0.25, label="Continual Stage 1 (raw)")
            w_c = max(1, min(50, len(c_rets) // 10))
            if len(c_rets) >= w_c:
                ax1.plot(c_cum_steps[w_c-1:], moving_average(c_rets, w_c), color="darkcyan", linewidth=2, label="Continual Stage 1 (Head 1)")
            has_c = True

    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Return")
    ax1.grid(True, linestyle="--", alpha=0.6)
    if has_b or has_c:
        ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Continual Transfer to Task 2 (Hopper-v4)
    ax2 = axes[1]
    ax2.set_title("Task 2 (Hopper-v4): Transfer Learning", fontweight="bold", fontsize=11)
    if continual_data is not None and len(continual_data.get("t2_returns", [])) > 0:
        t2_steps = continual_data["t2_steps"]
        t2_rets = continual_data["t2_returns"]
        t2_cum = np.cumsum(t2_steps)
        ax2.plot(t2_cum, t2_rets, color="orange", alpha=0.3, label="Hopper Raw Returns")
        w_t2 = max(1, min(50, len(t2_rets) // 10))
        if len(t2_rets) >= w_t2:
            ax2.plot(t2_cum[w_t2-1:], moving_average(t2_rets, w_t2), color="chocolate", linewidth=2, label="Continual Transfer (Head 2)")
        ax2.legend(loc="upper left", frameon=True)
    else:
        ax2.text(0.5, 0.5, "Stage 2 Transfer Data\nNot Yet Available", ha="center", va="center", color="gray", fontsize=12)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Episode Return")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Knowledge Retention & Forgetting Summary
    ax3 = axes[2]
    ax3.set_title("Catastrophic Forgetting Test (Task 1 Retention)", fontweight="bold", fontsize=11)
    if continual_data is not None and "t1_eval_before" in continual_data:
        ret_before = continual_data["t1_eval_before"][0]
        ret_after = continual_data["t1_eval_after"][0]
        retention = continual_data["retention_pct"]
        bars = ax3.bar(["Before Hopper\n(Stage 1 Final)", "After Hopper\n(Stage 3 Re-test)"], [ret_before, ret_after],
                       color=["#2b5c8f", "#d95f02" if retention < 70 else "#2ca02c"], width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + max(5, yval * 0.02), f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")
        ax3.set_ylabel("Deterministic Evaluation Return")
        ax3.set_title(f"Humanoid Retention: {retention:.1f}%", fontweight="bold", fontsize=11)
    else:
        ax3.text(0.5, 0.5, "Evaluation Summary\nNot Yet Available", ha="center", va="center", color="gray", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.6, axis="y")

    plt.tight_layout()
    output_png = "./results_comparison.png"
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"\nComparison plot updated and saved to: {output_png}")

    # 4. Print Summary Metrics Table
    print("\n------------------------------------------------------------------")
    print("METRICS BREAKDOWN:")
    print("------------------------------------------------------------------")
    if baseline_data is not None:
        b_steps, b_rets, _ = baseline_data
        print(f"Baseline Humanoid : Total Steps = {sum(b_steps):,}, Episodes = {len(b_rets):,}, Max Return = {max(b_rets):.2f}, Final 100 Ep Mean = {np.mean(b_rets[-100:]):.2f}")
    if continual_data is not None:
        t1_r = continual_data["t1_returns"]
        t2_r = continual_data["t2_returns"]
        print(f"Continual Humanoid: Total Steps = {sum(continual_data['t1_steps']):,}, Episodes = {len(t1_r):,}, Max Return = {max(t1_r):.2f}, Final 100 Ep Mean = {np.mean(t1_r[-100:]):.2f}")
        if len(t2_r) > 0:
            print(f"Continual Hopper  : Total Steps = {sum(continual_data['t2_steps']):,}, Episodes = {len(t2_r):,}, Max Return = {max(t2_r):.2f}, Final 100 Ep Mean = {np.mean(t2_r[-100:]):.2f}")
        if "retention_pct" in continual_data:
            print(f"Humanoid Retention: {continual_data['retention_pct']:.2f}% (Before: {continual_data['t1_eval_before'][0]:.2f} -> After: {continual_data['t1_eval_after'][0]:.2f})")
    print("------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
