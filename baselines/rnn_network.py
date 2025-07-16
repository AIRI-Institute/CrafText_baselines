import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict, Optional
import distrax
import functools

import logging
# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticTextVisualRNN(nn.Module):
    action_dim: int
    config: Dict
    layer_size: Optional[int] = None 

    @nn.compact
    def __call__(self, hidden, x, encoded_input):
        layer_size = self.layer_size if self.layer_size is not None else self.config['LAYER_SIZE']
        if layer_size is None:
            raise ValueError("LAYER_SIZE must be specified either in config or as a parameter.")

        obs, dones = x

        obs = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        obs = nn.relu(obs)
        obs = nn.max_pool(obs, window_shape=(3, 3), strides=(3, 3))
        obs = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        obs = nn.relu(obs)
        obs = nn.max_pool(obs, window_shape=(3, 3), strides=(3, 3))
        obs = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        obs = nn.relu(obs)
        obs = nn.max_pool(obs, window_shape=(3, 3), strides=(3, 3))

        obs_embedding = jnp.reshape(obs, (obs.shape[0], obs.shape[1], -1))

        obs_embedding = nn.Dense(
            layer_size // 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs_embedding)
        obs_embedding = nn.relu(obs_embedding)

        encoded_input = nn.Dense(
            layer_size // 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(encoded_input)
        encoded_input = nn.relu(encoded_input)


        combined_embedding = jnp.concatenate([obs_embedding, encoded_input], axis=-1)


        rnn_in = (combined_embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)


        actor_mean = nn.Dense(
            layer_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            layer_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)


        critic = nn.Dense(
            layer_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            layer_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)