import argparse
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import pygame
from craftax.craftax.renderer import render_craftax_pixels as render
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftext.environment.craftext_wrapper import InstructionWrapper 
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
import orbax.checkpoint as ocp
sys.path.append("./models")
from ..models.actor_critic import ActorCriticConv, ActorCriticWithEmbedding, ActorCritic
from ..models.actor_critic_with_text import (
                        ActorCriticConvWithBERT
)
import imageio
from craftax.craftax_env import make_craftax_env_from_name

os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    font = ImageFont.truetype("arial.ttf", 30)
except IOError:
    font = ImageFont.load_default()


def add_text_to_image(image, text):
    """Add text to an image."""
    text_to_list = text.split()
    text = ""
    for i in range(0, len(text_to_list), 6):
        text += " ".join(text_to_list[i:i+6])
        text += "\n"
        
    img_pil = Image.fromarray(image.astype(np.uint8))
    img_with_text = Image.new('RGB', (img_pil.width, img_pil.height + 50), color=(255, 255, 255))
    img_with_text.paste(img_pil, (0, 50))

    draw = ImageDraw.Draw(img_with_text)
    draw.text((10, 5), text, font=font, fill=(0, 0, 0))

    return np.array(img_with_text)


from PIL import Image

class CraftaxRenderer:
    def __init__(self, env, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.frames = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
        )
        if self.env.environment_key == 1:
            env_render = render
        else:
            env_render = render_classic
        self._render = jax.jit(env_render, static_argnums=(1,))

    def render(self, env_state):
        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        image = Image.fromarray(np.array(pixels).astype(np.uint8))
        self.frames.append(image)

    def render_to_image(self, env_state):
        """Render the environment state to an image array and resize it to 256x256."""
        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        
        # Convert pixels to image and resize to 256x256
        image = Image.fromarray(np.array(pixels).astype(np.uint8))
        resized_image = image.resize((256, 256))
        
        return np.array(resized_image)

    def save_gif(self, filename, duration=100):
        """Save the stored frames as a GIF."""
        if self.frames:
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )


def main(args):
    with open(os.path.join(args.path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

    config["NUM_ENVS"] = 1

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(
        os.path.abspath(os.path.join(args.path, "policies")), orbax_checkpointer, options
    )

    is_classic = False
    config_name = config["ENV_NAME"]

    add_text_emb = "-Tdext" in config_name
    config["ENV_NAME"] = config_name.replace("-Text", "")
    
    env = make_craftax_env_from_name(config["ENV_NAME"],  False)
    actions_count = 17 if "Classic" in config["ENV_NAME"] else 43
    if "Pixels" in config["ENV_NAME"]:
            network = ActorCriticConvWithBERT(actions_count, config["LAYER_SIZE"])
    else:
            network = ActorCritic(actions_count, config["LAYER_SIZE"])
                                    

    env = InstructionWrapper(env, args.craftext_settings)
    env_params = env.default_params

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)

    network_params = network.init(_rng, init_x, env.encoded_instruction)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(config["TOTAL_TIMESTEPS"])
    
   # network.apply(train_state['params'],init_x, env.encoded_instruction)

    obs, env_state = env.reset(_rng, env_params)
    done = False

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=1)
    steps = 0
    step_fn = jax.jit(env.step)
    observations = []
    while not done and steps < 500:
        obs = jnp.expand_dims(obs, axis=0)
        instruction = env.scenario_data.instructions_list[env_state.idx.item()]
        pi, value = network.apply(train_state['params'], obs, env_state.instruction.reshape(1, -1))
        action = pi.sample(seed=_rng)[0]
        
        action = jax.device_put(action, device=jax.devices('gpu')[0])

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            steps += 1

        image = renderer.render_to_image(env_state.env_state)
        observations.append(image)

    gif_name = instruction.replace(" ", "_")
    import random
    ix = random.randint(0,200)
    with imageio.get_writer(f'animation/{ix}_{gif_name}.gif', mode='I', duration=0.1) as writer:
        for i, image in enumerate(observations):
            text = f"Step {i}, Instruction {env.scenario_data.instructions_list[env_state.idx.item()]}"
            image_with_text = add_text_to_image(image, text)
            writer.append_data(image_with_text.astype(np.uint8))


def print_new_achievements(achievements_cls, old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{achievements_cls(i).name} ({new_achievements.sum()}/{22})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)