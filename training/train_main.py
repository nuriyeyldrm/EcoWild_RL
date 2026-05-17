import numpy as np
import json
import torch
import multiprocessing as mp
import os
import sys
import uuid
from datetime import datetime
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure

from data_preprocessing import load_and_prepare_data
from callbacks import (
    ResetPolicyOnHighSamplingCallback,
    ResetNoiseCallback,
)
from noise import AnnealedNormalActionNoise
from logger_utils import Logger


# -----------------------------
# Deterministic Settings
# -----------------------------
# Set deterministic CPU behavior
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# reduces nondeterminism from GPU kernel choices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------------
# Load Config
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python train_main.py <config_file.json>")
    sys.exit(1)

config_path = sys.argv[1]

with open(config_path, "r") as json_file:
    config = json.load(json_file)


# -----------------------------
# Global Parameter Extraction
# -----------------------------
# (KEEP EXACTLY AS BEFORE)
# Copy your parameter extraction block here
# Example:
TP_rate = config["ML_Performance"]["TP_rate"]
FP_rate = config["ML_Performance"]["FP_rate"]

reserved_energy = config["Energy_Constraints"]["reserved_energy"]
battery_energy = config["Initial_Battery_Levels"]

E_camera_host = config["Energy_Constraints"]["E_camera_host"]
E_comm = config["Energy_Constraints"]["E_comm"]
E_proc_ml = config["Energy_Constraints"]["E_proc_ml"]
E_proc_rl = config["Energy_Constraints"]["E_proc_rl"]
E_temp_humidity_sensor = config["Energy_Constraints"]["E_temp_humidity_sensor"]
E_anemometer_sensor = config["Energy_Constraints"]["E_anemometer_sensor"]

# Standby power consumption
P_temp_humidity_standby = config["Standby_Power_Components"]["P_temp_humidity_standby"]
P_anemometer_standby = config["Standby_Power_Components"]["P_anemometer_standby"]
P_camera_standby = config["Standby_Power_Components"]["P_camera_standby"]
P_comm_standby = config["Standby_Power_Components"]["P_comm_standby"]

E_battery_leakage_percentage = config["E_battery_leakage_percentage"]
harvested_energy_loss = config["harvested_energy_loss"]

beta = config["Reward_Params"]["beta"]
k1 = config["Reward_Params"]["k1"]
alpha_B = config["Reward_Params"]["alpha_B"]
R_min = config["Reward_Params"]["R_min"]

gamma_x = config["TD3_params"]["gamma"]
tau_x = config["TD3_params"]["tau"]
learning_rate_x = config["TD3_params"]["learning_rate"]

total_timesteps_x = config["TD3_params"]["total_timesteps"]
batch_size_x = config["TD3_params"]["batch_size"]
buffer_size_x = config["TD3_params"]["buffer_size"]

target_decay_step_x = config["TD3_params"]["target_decay_step"]
decay_type_x = config["TD3_params"]["decay_type"]

max_missing_fire_min = config["max_missing_fire_min"]

min_sampling_time = config["TD3_params"]["min_sampling_time"]
max_sampling_time = config["TD3_params"]["max_sampling_time"]

sigma_start_x = config["TD3_params"]["sigma_start"] * 2 / (max_sampling_time - min_sampling_time) / 4 # 4: gaussian 4\sigma X Scale <= Allowed change
sigma_end_x = config["TD3_params"]["sigma_end"] * 2 / (max_sampling_time - min_sampling_time) / 4 # 4: gaussian 4\sigma X Scale <= Allowed change

file_name = config["file_name"]


# -----------------------------
# Load Dataset
# -----------------------------
df = load_and_prepare_data(config)


# Import WildfireEnv AFTER df exists
from wildfire_env import WildfireEnv


# -----------------------------
# Training Function
# -----------------------------
def run_training_with_seed(custom_seed):

    torch.manual_seed(custom_seed)
    np.random.seed(custom_seed)

    folder = f"Data/episode_plots_step_reward{file_name}_{custom_seed}"
    os.makedirs(folder, exist_ok=True)

    log_file_path = f"{folder}/wildfire_rl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    env = WildfireEnv(df, config)
    obs = env.reset()

    log_dir = f"{folder}/tensorboard_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_step_reward{file_name}/"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["tensorboard"])

    action_noise_x = AnnealedNormalActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma_start=sigma_start_x,
        sigma_end=sigma_end_x,
        total_timesteps=total_timesteps_x,
        target_decay_step=target_decay_step_x,
        decay_type=decay_type_x
    )

    model = TD3(
        "MlpPolicy",
        env,
        seed=custom_seed,
        learning_rate=learning_rate_x,
        buffer_size=buffer_size_x,
        batch_size=batch_size_x,
        gamma=gamma_x,
        tau=tau_x,
        action_noise=action_noise_x,
        verbose=1,
        tensorboard_log=log_dir,
        # Optional:
        # learning_starts=learning_starts_x,
        # policy_delay=3,
    )

    model.set_logger(logger)
    model.learn(
        total_timesteps=total_timesteps_x,
        log_interval=1,
        callback=[
            ResetNoiseCallback(action_noise_x),
            ResetPolicyOnHighSamplingCallback(threshold=55, required_streak=4),
        ],
    )

    model.save(f"{folder}/wildfire_td3_final")

    np.save(f"{folder}/rewards_seed_{custom_seed}.npy", env.cumulative_rewards_list)
    np.save(f"{folder}/average_rewards_seed_{custom_seed}.npy", env.average_rewards_list)
    np.save(f"{folder}/tensorboard_seed_{custom_seed}.npy", env.tensorboard_rewards_list)
    np.save(f"{folder}/avg_sampling_time_seed_{custom_seed}.npy", env.avg_sampling_time_list)
    np.save(f"{folder}/detection_time_seed_{custom_seed}.npy", env.detection_time_list)
    
    step_log_array = np.array(sorted(env.step_reward_log, key=lambda x: x[0]))
    step_sampling_log_array = np.array(sorted(env.step_sampling_log, key=lambda x: x[0]))
    np.save(f"{folder}/step_rewards_seed_{custom_seed}.npy", step_log_array)
    np.save(f"{folder}/step_sampling_time_seed_{custom_seed}.npy", step_sampling_log_array)

    model.save(f"{folder}/wildfire_td3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Training finished with seed {custom_seed}")


# -----------------------------
# Multiprocessing Entry
# -----------------------------
if __name__ == "__main__":

    file_name = config["file_name"]
    folder = f"Plot/episode_plots_step_reward{file_name}"
    os.makedirs(folder, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    seeds = [int(uuid.uuid4().int % 1e8) for _ in range(5)]
    processes = []

    for s in seeds:
        p = mp.Process(target=run_training_with_seed, args=(s,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Process {p.name} failed with exit code {p.exitcode}")

    def plot_rl_style_all_metrics(seeds, config, bin_count=100, total_timesteps=total_timesteps_x): #TODO
        metrics = [
            ("rewards", "Step Reward"),
            ("sampling_time", "Step Sampling Time (min)"),
        ]

        fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 16))

        for idx, (metric_name, ylabel) in enumerate(metrics):
            all_x, all_y = [], []
            for s in seeds:
                path = f"Data/episode_plots_step_reward{config['file_name']}_{s}/step_{metric_name}_seed_{s}.npy"
                if os.path.exists(path):
                    data = np.load(path)
                    all_x.extend(data[:, 0])  # timesteps
                    all_y.extend(data[:, 1])  # rewards
                else:
                    print(f"Missing file for seed {s}: {metric_name}")

            # Convert to numpy arrays
            all_x, all_y = np.array(all_x), np.array(all_y)

            # Binning by timesteps
            bins = np.linspace(0, total_timesteps, bin_count + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            means, stds = [], []

            for i in range(bin_count):
                mask = (all_x >= bins[i]) & (all_x < bins[i + 1])
                bin_data = all_y[mask]
                if bin_data.size > 0:
                    means.append(np.mean(bin_data))
                    stds.append(np.std(bin_data))
                else:
                    means.append(np.nan)
                    stds.append(0)

            axs[idx].plot(bin_centers, means, label=ylabel, color="green")
            axs[idx].fill_between(bin_centers, np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color="green")
            axs[idx].set_ylabel(ylabel, fontsize=18, fontweight='bold')
            axs[idx].legend(fontsize=18)
            axs[idx].tick_params(axis='both', labelsize=18)
            axs[idx].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        axs[-1].set_xlabel("Timesteps", fontsize=18, fontweight='bold')
        fig.suptitle("Per-Step Reward and Sampling Time Trends Over Timesteps", fontsize=18, fontweight='bold', y=0.99)
        fig.align_ylabels(axs)  # Align all y-axis labels across subplots
        fig.tight_layout(pad=1.5)  # Smaller padding between subplots
        plt.savefig(f"{folder}/comparison_rl_style_all_metrics{config['file_name']}.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("Saved: comparison_rl_style_all_metrics.png (RL-style)")

        
    def plot_all_metrics_across_seeds():
        fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

        metrics = [
            ("rewards", "Cumulative Reward"),
            ("average_rewards", "Average Reward"),
            ("tensorboard", "Tensorboard Reward"),
            ("avg_sampling_time", "Average Sampling Time (min)"),
            ("detection_time", "Detection Time (min)")
        ]

        for idx, (metric_name, ylabel) in enumerate(metrics):
            for s in seeds:
                path = f"Data/episode_plots_step_reward{config['file_name']}_{s}/{metric_name}_seed_{s}.npy"
                if os.path.exists(path):
                    data = np.load(path)
                    axs[idx].plot(data, label=f"Seed {s}", linewidth=2)
                else:
                    print(f"Missing file for seed {s}: {metric_name}")
            axs[idx].set_ylabel(ylabel, fontsize=18, fontweight='bold')
            axs[idx].legend(fontsize=18)
            axs[idx].tick_params(axis='both', labelsize=18, width=2)
            axs[idx].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        axs[-1].set_xlabel("Episode", fontsize=18, fontweight='bold')
        fig.suptitle("Comparison of Metrics Across Seeds", fontsize=18, fontweight='bold', y=0.95)
        fig.align_ylabels(axs)  # Align all y-axis labels across subplots
        fig.tight_layout(pad=1.5)  # Smaller padding between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f"{folder}/comparison_all_metrics{config['file_name']}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved combined plot: comparison_all_metrics.png")
        
    def plot_all_metrics_across_seeds_modified():
        fig, axs = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

        metrics = [
            ("average_rewards", "Average Reward"),
            ("avg_sampling_time", "Avg. Sampling Time (min)"),
            ("detection_time", "Detection Time (min)")
        ]

        for idx, (metric_name, ylabel) in enumerate(metrics):
            for s in seeds:
                path = f"Data/episode_plots_step_reward{config['file_name']}_{s}/{metric_name}_seed_{s}.npy"
                if os.path.exists(path):
                    data = np.load(path)
                    axs[idx].plot(data, label=f"Seed {s}", linewidth=2)
                else:
                    print(f"Missing file for seed {s}: {metric_name}")
            axs[idx].set_ylabel(ylabel, fontsize=18, fontweight='bold')
            axs[idx].tick_params(axis='both', labelsize=18, width=2)
            axs[idx].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        axs[-1].set_xlabel("Episode", fontsize=18, fontweight='bold')
        fig.suptitle("Comparison of Metrics Across Seeds", fontsize=18, fontweight='bold', y=0.95)
        fig.align_ylabels(axs)  # Align all y-axis labels across subplots
        fig.tight_layout(pad=1.5)  # Smaller padding between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f"{folder}/comparison_all_metrics{config['file_name']}_modified.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved combined plot: comparison_all_metrics.png")


    plot_all_metrics_across_seeds()
    plot_all_metrics_across_seeds_modified()
    plot_rl_style_all_metrics(seeds, config)