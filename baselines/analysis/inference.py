import argparse
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import pandas as pd
import imageio
import random

from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
sys.path.append("./models")
from ..models.actor_critic import ActorCriticConv, ActorCriticWithEmbedding
from ..models.actor_critic_with_text import (
                        ActorCriticConvWithBERT
)
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper 
from baselines.analysis.view_ppo_agent import CraftaxRenderer, add_text_to_image
sys.path.append(".")
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper)


class ResultManager:
    def __init__(self, experiment_name, craftext_settings):
        self.experiment_name = experiment_name
        self.craftext_settings = craftext_settings
        self.instructions = None
        self.functions = None
        self.success_rates = None

    def update_results(self, instructions, functions, success_rates):
        self.instructions = instructions
        self.functions = functions
        self.success_rates = success_rates

    def save_to_csv(self, output_path):
        print(f"Results saved to {output_path}")
        if self.instructions is None or self.functions is None or self.success_rates is None:
            raise ValueError("Results are not initialized.")
        
        dataset = pd.DataFrame({
            'instructions': self.instructions,
            'functions': self.functions,
            'sr': self.success_rates
        })
        dataset.to_csv(output_path, index=False)



class Experiment:
    def __init__(self, args):
        self.args = args
        self.config = self._load_config()
        self.checkpoint_manager = self._initialize_checkpoint_manager()
        self.env, self.network = self._initialize_environment_and_network(view=args.view)
        self.train_state = self._initialize_train_state()
        self.result_manager = ResultManager(args.experiment_name, args.craftext_settings)

    def _load_config(self):
        config_path = os.path.join(self.args.path, "config.yaml")
        with open(config_path) as f:
            raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {key: value["value"] if isinstance(value, dict) and "value" in value else value
                  for key, value in raw_config.items()}
        config["NUM_ENVS"] = self.args.num_envs
        config["RATIO"] = self.args.ratio
        return config

    def _initialize_checkpoint_manager(self):
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_path = os.path.abspath(os.path.join(self.args.path, "checkpoint_restart_1"))
        return CheckpointManager(checkpoint_path, orbax_checkpointer, options)

    def _initialize_environment_and_network(self, view=False):
        is_classic = "-Text" not in self.config["ENV_NAME"]
        env_name = self.config["ENV_NAME"].replace("-Text", "")
        self.config["ENV_NAME"] = env_name

        env = make_craftax_env_from_name(env_name, False)
        actions_count = 17 if "Classic" in env_name else 43
        network_class = ActorCriticConvWithBERT if "Pixels" in env_name else ActorCriticConv
        network = network_class(actions_count, self.config["LAYER_SIZE"])

        env = InstructionWrapper(env, self.args.craftext_settings)
        if not view:
            env = OptimisticResetVecEnvWrapper(env, self.config["NUM_ENVS"], 
                                            min(self.config["RATIO"], self.config["NUM_ENVS"]))
        return env, network

    def _initialize_train_state(self):
        init_x = jnp.zeros((self.config["NUM_ENVS"], *self.env.observation_space(self.env.default_params).shape))
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        rng, _rng, __rng = jax.random.split(rng, 3)

        instructions = jnp.tile(self.env.encoded_instruction, (self.config["NUM_ENVS"], 1))
        network_params = self.network.init(_rng, init_x, instructions)

        tx = optax.chain(
            optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
            optax.adam(self.config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
        )
        return self.checkpoint_manager.restore(4*int(self.config["TOTAL_TIMESTEPS"]))
    
    def view(self):
        rng = jax.random.PRNGKey(random.randint(0, 100))
        obs, env_state = self.env.reset(rng, self.env.default_params)
        step_fn = jax.jit(self.env.step)
        done = False
        renderer = CraftaxRenderer(self.env, self.env.default_params, pixel_render_size=1)
        steps = 0
        step_fn = jax.jit(self.env.step)
        params = self.train_state['runner_state'][0]["params"]
        observations = []
        while not done and steps < 500:
            obs = jnp.expand_dims(obs, axis=0)
            instruction =self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]
            pi, value = self.network.apply(params, obs, env_state.instruction.reshape(1, -1))
            action = pi.sample(seed=rng)[0]
            
            action = jax.device_put(action, device=jax.devices('gpu')[0])

            if action is not None:
               # rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(
                    rng, env_state, action, self.env.default_params
                )
                steps += 1

            image = renderer.render_to_image(env_state.env_state)
            observations.append(image)
        gif_name = "_".join(self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()].split()[:5])
        ix = random.randint(0,200)
        folder_name = "animation"

        os.makedirs(folder_name, exist_ok=True)
        with imageio.get_writer(f'animation/{ix}_{gif_name}.gif', mode='I', duration=0.1) as writer:
            for i, image in enumerate(observations):
                text = f"Step {i}, Instruction {self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]}"
                image_with_text = add_text_to_image(image, text)
                writer.append_data(image_with_text.astype(np.uint8))
        print(f'Save with name animation/{ix}_{gif_name}.gif')

    def run(self):
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        obs, env_state = self.env.reset(rng, self.env.default_params)

        steps = 0
        step_fn = jax.jit(self.env.step)
        total_success_rate = np.zeros(self.config["NUM_ENVS"])
        done_count = np.zeros(self.config["NUM_ENVS"])
        prev_indx = np.zeros(self.config["NUM_ENVS"])
        params = self.train_state['runner_state'][0]["params"]

        while steps < 5000:
            pi, value = self.network.apply(params, obs, env_state.instruction)
            action = pi.sample(seed=rng)
            if action is not None:
                rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(_rng, env_state, action, self.env.default_params)
                steps += 1

                instruction_done_float = info['SR']
                indices = np.where(instruction_done_float > 0)
                for inst in prev_indx[indices]:
                    total_success_rate[inst] += 1

                done_indices = np.where(done > 0)
                for inst in prev_indx[done_indices]:
                    done_count[inst] += 1

                prev_indx = env_state.idx

        success_rates = total_success_rate / done_count
        self.result_manager.update_results(
            self.env.scenario_handler.scenario_data.instructions_list,
            self.env.scenario_handler.scenario_data.str_check_lambda_list,
            success_rates[:len(self.env.scenario_handler.scenario_data.instructions_list)]
        )
        os.makedirs("results", exist_ok=True)
        self.result_manager.save_to_csv(f"results/{self.args.experiment_name}_{self.args.craftext_settings}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--view", type=bool, default=False)
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--ratio", type=int, default=16)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if args.path is None:
        args.path = f"./wandb/{args.experiment_name}/files/"
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    experiment = Experiment(args)
    if args.view:
        experiment.view()
    else:
        if args.debug:
            with jax.disable_jit():
                experiment.run()
        else:
            experiment.run()