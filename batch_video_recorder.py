import os
import sys
import time
import gymnasium as gym

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from simulate import run_mujoco

def main():
    print("==================================================================")
    print("       BATCH VIDEO RECORDING: Continual (Stage 1 & 2) & Baseline   ")
    print("==================================================================")
    start_all = time.time()

    jobs = [
        {
            "name": "Stage 1 Continual (Humanoid-v4, 100 Episodes)",
            "env": "Humanoid-v4",
            "model": "results_continual/20261020_162439-ContinualAVG_Humanoid-v4_to_Hopper-v4_seed42_stage1_final.pt",
            "episodes": 100,
            "out": "results_continual/stage1_humanoid_100ep_video.mp4",
            "plot": "results_continual/stage1_humanoid_100ep_video_plot.png",
        },
        {
            "name": "Stage 2 Continual (Hopper-v4, 100 Episodes)",
            "env": "Hopper-v4",
            "model": "results_continual/20261020_162439-ContinualAVG_Humanoid-v4_to_Hopper-v4_seed42_stage2_final.pt",
            "episodes": 100,
            "out": "results_continual/stage2_hopper_100ep_video.mp4",
            "plot": "results_continual/stage2_hopper_100ep_video_plot.png",
        },
        {
            "name": "Baseline Monolithic (Humanoid-v4, 500 Episodes)",
            "env": "Humanoid-v4",
            "model": "results_baseline/20261020_161319-AVG-Humanoid-v4_seed-42_model.pt",
            "episodes": 500,
            "out": "results_baseline/baseline_humanoid_500ep_video.mp4",
            "plot": "results_baseline/baseline_humanoid_500ep_video_plot.png",
        },
    ]

    for idx, job in enumerate(jobs, 1):
        print(f"\n[{idx}/3] Starting {job['name']}...")
        t0 = time.time()
        os.makedirs(os.path.dirname(job['out']) or ".", exist_ok=True)
        
        run_mujoco(
            env_name=job['env'],
            model_path=job['model'],
            episodes=job['episodes'],
            render_mode="record",
            output_file=job['out'],
            plot=True,
            plot_file=job['plot'],
            action_mode="deterministic"
        )
        
        t1 = time.time()
        file_sz = os.path.getsize(job['out']) / (1024 * 1024) if os.path.exists(job['out']) else 0
        print(f"[{idx}/3] Finished {job['name']} in {t1-t0:.1f}s | Output: {job['out']} ({file_sz:.2f} MB)")

    total_time = time.time() - start_all
    print("\n==================================================================")
    print(f"All 3 video recordings finished successfully in {total_time:.1f}s ({total_time/60:.1f} min)!")
    print("==================================================================")

if __name__ == "__main__":
    main()
