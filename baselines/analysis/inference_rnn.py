import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import yaml
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)
from tqdm import tqdm

sys.path.append("./models")
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.encoders.craftext_distilbert_model_encoder import make_encoder
from craftext.environment.scenarious.manager import create_scenarios_with_dataset
from rnn_network import ActorCriticTextVisualRNN, ScannedRNN

sys.path.append(".")
from wrappers import OptimisticResetVecEnvWrapper


def parse_function_name(function_str):
    # Step 1: Split by the first opening parenthesis and take the left part
    left_part = function_str.split("(", 1)[0]
    if len(left_part) <= 1:
        return left_part

    # Step 2: Safely split by space, colon, or quotes, and handle gracefully
    try:
        function_name = (
            left_part.split()[-1]  # Take the last word from space split
            .split(":")[-1]  # Split by colon and take the last part
            .split('"')[-1]  # Split by double quotes
            .split("'")[-1]  # Split by single quotes
        )
        return function_name
    except IndexError:
        return "Error: Invalid format"


class ResultManager:
    def __init__(self, experiment_name, craftext_settings):
        self.experiment_name = experiment_name
        self.craftext_settings = craftext_settings
        self.instructions = None
        self.functions = None
        self.function_names = None
        self.success_rates = None

    def update_results(self, instructions, functions, success_rates):
        self.instructions = instructions
        # There is a bug, each element if functions is a list with identical value
        self.functions = [
            f[0] if isinstance(f, (list, tuple)) else f for f in functions
        ]
        self.function_names = [
            parse_function_name(f) if f is not None else None for f in self.functions
        ]
        self.success_rates = success_rates

    def save_to_csv(self, output_path):
        if (
            self.instructions is None
            or self.functions is None
            or self.success_rates is None
        ):
            raise ValueError("Results are not initialized.")

        dataset = pd.DataFrame(
            {
                "instructions": self.instructions,
                "functions": self.functions,
                "functions_name": self.function_names,
                "sr": self.success_rates,
            }
        )
        overall_mean_sr = dataset["sr"].mean()
        mean_sr_dict = dataset.groupby("functions_name")["sr"].mean().to_dict()
        mean_sr_dict["overall_mean_sr"] = overall_mean_sr

        dataset.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        return dataset, mean_sr_dict


class ExperimentArgs:
    def __init__(
        self,
        num_envs,
        experiment_name,
        ratio,
        checkpoint_num,
        env_name,
        max_grad_norm,
        lr,
        layer_size,
        total_timesteps,
        craftext_settings,
        path,
        view,
        use_plans,
        inference_step,
        expand_emb,
    ):
        self.num_envs = num_envs
        self.experiment_name = experiment_name
        self.ratio = ratio
        self.checkpoint_num = checkpoint_num
        self.env_name = env_name
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.layer_size = layer_size
        self.total_timesteps = total_timesteps
        self.craftext_settings = craftext_settings
        self.path = path
        self.view = view
        self.use_plans = use_plans
        self.inference_step = inference_step
        self.expand_emb = expand_emb


def experiment_args_from_config(args):
    config_path = os.path.join(args.path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Load the configuration file
    with open(config_path, "r") as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

    # Extract parameters from both raw_config and args

    env_name = raw_config.get("ENV_NAME", "default_env")["value"]  # Default if missing
    max_grad_norm = raw_config.get("MAX_GRAD_NORM", 0.5)[
        "value"
    ]  # Default max gradient norm
    lr = raw_config.get("LR", 0.001)["value"]  # Default learning rate
    total_timesteps = raw_config.get("TOTAL_TIMESTEPS", 1_000_000)[
        "value"
    ]  # Default timesteps
    layer_size = raw_config.get("LAYER_SIZE", 512)["value"]
    craftext_settings = getattr(
        args, "craftext_settings", None
    )  # Handle optional arguments
    num_envs = args.num_envs
    ratio = args.ratio
    experiment_name = args.experiment_name
    checkpoint_num = args.checkpoint_num
    use_plans = args.use_plans
    inference_step = args.inference_step
    expand_emb = args.expand_emb
    path = getattr(args, "path", None)
    view = getattr(args, "view", None)

    # Create and return the ExperimentArgs object
    experiment_args = ExperimentArgs(
        num_envs=num_envs,
        experiment_name=experiment_name,
        ratio=ratio,
        checkpoint_num=checkpoint_num,
        env_name=env_name,
        max_grad_norm=max_grad_norm,
        lr=lr,
        layer_size=layer_size,
        total_timesteps=total_timesteps,
        craftext_settings=craftext_settings,
        path=path,
        view=view,
        use_plans=use_plans,
        inference_step=inference_step,
        expand_emb=expand_emb,
    )

    return experiment_args


class Experiment:
    def __init__(self, args):
        # self.args = args
        self.config = args  # self._load_config()
        self.checkpoint_manager = self._initialize_checkpoint_manager()
        self.env, self.network = self._initialize_environment_and_network(
            view=self.config.view
        )
        self.train_state = self._initialize_train_state()
        self.result_manager = ResultManager(
            self.config.experiment_name, self.config.craftext_settings
        )

    def _initialize_checkpoint_manager(self):
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_path = os.path.abspath(self.config.path)
        return CheckpointManager(checkpoint_path, orbax_checkpointer, options)

    def _initialize_environment_and_network(self, view=False):
        env_name = self.config.env_name.replace("-Text", "")
        self.config.env_name = env_name

        env = make_craftax_env_from_name(env_name, False)
        actions_count = 17 if "Classic" in env_name else 43
        network_class = ActorCriticTextVisualRNN
        network = network_class(
            actions_count, self.config, layer_size=self.config.layer_size
        )

        if self.config.use_plans:
            encoder = make_encoder(n_splits=5)
            scenarious_loader = create_scenarios_with_dataset(True)
            env = InstructionWrapper(
                env,
                self.config.craftext_settings,
                encode_model_class=encoder,
                scenario_handler_class=scenarious_loader,
            )
        else:
            encoder = make_encoder(n_splits=self.config.expand_emb)
            env = InstructionWrapper(
                env, self.config.craftext_settings, encode_model_class=encoder
            )
        if not view:
            env = OptimisticResetVecEnvWrapper(
                env, self.config.num_envs, min(self.config.ratio, self.config.num_envs)
            )
        return env, network

    def _initialize_train_state(self):
        """Инициализация состояния сети и загрузка чекпоинта."""
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        rng, _rng = jax.random.split(rng)
        # Оптимизатор
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.lr, eps=1e-5),
        )
        # Загрузка чекпоинта
        train_state = self.checkpoint_manager.restore(int(self.config.total_timesteps))
        network_params = train_state["runner_state"][0]["params"]

        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
        )

        return train_state

    def run(self):
        """Инференс для сети с RNN."""
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        obs, env_state = self.env.reset(rng, self.env.default_params)

        steps = 0
        step_fn = jax.jit(self.env.step)
        hidden_state = ScannedRNN.initialize_carry(
            self.config.num_envs, self.config.layer_size
        )
        done = jnp.zeros((self.config.num_envs,), dtype=bool)

        num_instructions = len(
            self.env.scenario_handler.scenario_data.instructions_list
        )
        total_success_rate = np.zeros(num_instructions)
        done_count = np.zeros(num_instructions)

        prev_indx = jnp.zeros(self.config.num_envs, dtype=bool)

        steps = 0
        step_fn = jax.jit(self.env.step)

        #  params = self.train_state['runner_state'][0]["params"]
        total_steps = self.config.inference_step
        with tqdm(total=total_steps, desc="Simulation Steps") as pbar:
            while steps < total_steps:
                # Prepare RNN input
                rnn_input = (obs[np.newaxis, :], done[np.newaxis, :])

                # RNN inference
                hidden_state, pi, _ = self.network.apply(
                    self.train_state.params,
                    hidden_state,
                    rnn_input,
                    env_state.instruction[np.newaxis, :],
                )
                action = pi.sample(seed=rng).squeeze(0)
                rng, _rng = jax.random.split(rng)

                # Environment step
                obs, env_state, reward, done, info = step_fn(
                    rng, env_state, action, self.env.default_params
                )
                steps += 1

                # Update success rates
                instruction_done_float = info["SR"]
                indices = np.where(instruction_done_float > 0)
                for inst in prev_indx[indices]:
                    # print(inst)
                    total_success_rate[int(inst)] += 1

                # Update done counts
                done_indices = np.where(done > 0)
                for inst in prev_indx[done_indices]:
                    done_count[int(inst)] += 1

                # Update previous indices
                prev_indx = env_state.idx

                # Update progress bar
                pbar.update(1)

        success_rates = total_success_rate / done_count
        self.result_manager.update_results(
            self.env.scenario_handler.scenario_data.instructions_list,
            self.env.scenario_handler.scenario_data.str_check_lambda_list,
            success_rates[
                : len(self.env.scenario_handler.scenario_data.instructions_list)
            ],
        )

        evaluation_dir = os.path.join(self.config.experiment_name, "evaluation")
        os.makedirs(evaluation_dir, exist_ok=True)
        return self.result_manager.save_to_csv(
            f"{evaluation_dir}/{self.config.craftext_settings}_{self.config.total_timesteps // int(1e6)}M.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--view", type=bool, default=False)
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments"
    )
    parser.add_argument("--ratio", type=int, default=16)
    parser.add_argument("--checkpoint_num", type=str, default="checkpoint_restart_1")
    parser.add_argument("--use_plans", type=bool, default=False)
    parser.add_argument("--inference_step", type=int, default=2000)
    parser.add_argument("--expand_emb", type=int, default=1)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if args.path is None:
        args.path = f"./wandb/{args.experiment_name}/files/"
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    experiment_args = experiment_args_from_config(args)
    experiment = Experiment(experiment_args)
    if args.view:
        experiment.view()
    else:
        if args.debug:
            with jax.disable_jit():
                experiment.run()
        else:
            experiment.run()