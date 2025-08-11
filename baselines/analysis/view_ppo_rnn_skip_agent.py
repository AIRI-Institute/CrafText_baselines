import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)
from PIL import ImageFont

import imageio
from craftax.craftax_env import make_craftax_env_from_name

sys.path.append("./baselines")
sys.path.append("./models")
sys.path.append(".")
sys.path.append("..")

from rnn_network import ActorCriticTextVisualRNN, ScannedRNN
from analysis.render import CraftaxRenderer, add_text_to_image

os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    font = ImageFont.truetype("arial.ttf", 30)
except (ImportError, IOError):
    font = ImageFont.load_default()


def main(args):

    config = {
        "env_name": "Craftax-Classic-Pixels-v1",
        "craftext_settings": args.craftext_settings,
        'super_dataset': "/home/n.sorokin/SuperIgor-nsorokin-baselines/super_experiments/resources/temp_dataset/super_dataset0.json",
        # embeddings
        "embedding_source": 0,
        "encode_form_name": "EMBEDDING",
        "encode_form": EncodeForm.EMBEDDING,
    } 

    config = {k.upper(): v for k, v in config.items()}


    config["NUM_ENVS"] = 1

    config["RUN_NAME"] = "PPO-T-Baseline-test_20250801-134443"

    run_name = config["RUN_NAME"]

    checkpoint_dir = os.path.abspath(
        os.path.join("./experiments", run_name, "checkpoints")
    )
    os.makedirs(checkpoint_dir, exist_ok=True)  # Важно создать папку

    orbax_checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,  # Используем новый, чистый путь
        orbax_checkpointer,
        CheckpointManagerOptions(),
    )

    # Rest of your setup code...
    config_name = config["ENV_NAME"]
    config["ENV_NAME"] = config_name.replace("-Text", "")
    env = make_craftax_env_from_name(config["ENV_NAME"], False)

    env = InstructionWrapper(env, config["CRAFTEXT_SETTINGS"])
    env_params = env.default_params

    network = ActorCriticTextVisualRNN(
        env.action_space(env_params).n, config, layer_size=512
    )

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)

    config["CURRENT_TIMESTEP"] = 512

    train_state = checkpoint_manager.restore(config["CURRENT_TIMESTEP"])

    obs, env_state = env.reset(_rng, env_params)
    done = False

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=1)
    steps = 0
    step_fn = jax.jit(env.step)
    observations = []
    sum_rewards = []
    success_rates = []
    sum_reward = 0
    
    hidden_state_executor = ScannedRNN.initialize_carry(
        batch_size=1, hidden_size=512
    )

    done = jnp.zeros((1,), dtype=bool)

    while not done and steps < 500:
        obs = jnp.expand_dims(obs, axis=0)
        instruction = env.scenario_handler.scenario_data.instructions_list[
            env_state.idx.item()
        ]

        ac_in = (obs[np.newaxis, np.newaxis, :], done[np.newaxis, :])
        def_instruction = env_state.instruction[np.newaxis, np.newaxis, :]

        hidden_state, pi, _ = network.apply(
            train_state["runner_state"][0]["params"],
            hidden_state_executor,
            ac_in,
            def_instruction,
        )

        # action = pi.sample(seed=_rng)[0]
        action = jnp.squeeze(pi.sample(seed=rng))

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            steps += 1
            
            done = jnp.array([done])

            sum_reward += reward
            sum_rewards.append(sum_reward)

        image = renderer.render_to_image(env_state.env_state)
        observations.append(image)
        print(info)
        success_rates.append(info.get("SR", 0.0))

    gif_name = instruction.replace(" ", "_")
    import random

    os.makedirs(f"./runs/{config['RUN_NAME']}/animation_{config['CRAFTEXT_SETTINGS']}", exist_ok=True)

    ix = random.randint(0, 200)
    with imageio.get_writer(
        f"./runs/{config['RUN_NAME']}/animation_{config['CRAFTEXT_SETTINGS']}/{gif_name}_{ix}.gif",
        mode="I",
        duration=0.3,
    ) as writer:
        for i, image in enumerate(observations):
            text = f"Step {i}, Instruction {env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]}, Reward {sum_rewards[i]:.2f}, SR {success_rates[i]:.2f}"
            image_with_text = add_text_to_image(image, text)
            writer.append_data(image_with_text.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--current_timestep", type=int, default=512)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    print("args parsed")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
