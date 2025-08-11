import os

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    OBS_DIM,
)
from craftax.craftax.renderer import render_craftax_pixels as render
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from PIL import Image, ImageDraw, ImageFont

os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    font = ImageFont.truetype("arial.ttf", 30)
except (ImportError, IOError):
    font = ImageFont.load_default()


def add_text_to_image(image, text):
    """Add text to an image."""
    text_to_list = text.split()
    text = ""
    for i in range(0, len(text_to_list), 6):
        text += " ".join(text_to_list[i : i + 6])
        text += "\n"

    img_pil = Image.fromarray(image.astype(np.uint8))
    img_with_text = Image.new(
        "RGB", (img_pil.width, img_pil.height + 50), color=(255, 255, 255)
    )
    img_with_text.paste(img_pil, (0, 50))

    draw = ImageDraw.Draw(img_with_text)
    draw.text((10, 5), text, font=font, fill=(0, 0, 0))

    return np.array(img_with_text)


class CraftaxRenderer:
    def __init__(self, env, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.frames = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
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

    def save_gif(self, filename, duration=300):
        """Save the stored frames as a GIF."""
        if self.frames:
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0,
            )