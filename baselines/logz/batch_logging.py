import time

import jax.numpy as jnp
import numpy as np

batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
         "SR": info["SR"],
    }   

    sum_achievements = 0
    for k, v in info.items():
        if "achievements" in k.lower():
            to_log[k] = v
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log


def batch_log_tb(update_step, log, config, writer):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config.get('NUM_REPEATS', 1):
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                continue
            else:
                for i in range(config.get('NUM_REPEATS', 1)):
                    val = batch_logs[update_step][i][key]
                    if not jnp.isnan(val):
                        agg.append(val)

            if len(agg) > 0:
                # Ваша логика агрегации сохранена
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                    "SR",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg).mean()

        log_times.append(time.time())

        if config['DEBUG']:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                     config['NUM_STEPS'] * config['NUM_ENVS'] * config.get('NUM_REPEATS', 1)
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        if writer:
            for key, value in agg_logs.items():
                if isinstance(value, (np.ndarray, list)):
                    value_mean = np.mean(value)
                else:
                    value_mean = value
                
                if not np.isnan(value_mean):
                    writer.scalar(f"train_live/{key}", value_mean, update_step)
        
        del batch_logs[update_step]
