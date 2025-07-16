import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

import distrax

from typing import Sequence, List, Callable

import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, List, Sequence, Tuple, Optional
from flax.linen.initializers import orthogonal, constant
import distrax


# Impala Baseline -----------------------------

def get_nonlinearity(name: str):
    if name == "elu":
        return nn.elu
    elif name == "relu":
        return nn.relu
    elif name == "tanh":
        return nn.tanh
    else:
        raise ValueError(f"Unknown activation function: {name}")


class MLP(nn.Module):
    layer_sizes: List[int]
    activation_fn: Callable = nn.relu  # Default nonlinearity is relu

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation_fn(x)
        return x


class ResBlock(nn.Module):
    channels: int
    nonlinearity: str

    @nn.compact
    def __call__(self, x):
        act = get_nonlinearity(self.nonlinearity)
        residual = x
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = act(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        return act(x + residual)


class ResnetImpalaEncoder(nn.Module):
    nonlinearity: str
    mlp_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = get_nonlinearity(self.nonlinearity)
        resnet_conf = [[16, 2], [32, 2], [32, 2]]
        for out_ch, num_blocks in resnet_conf:
            x = nn.Conv(out_ch, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
            for _ in range(num_blocks):
                x = ResBlock(channels=out_ch, nonlinearity=self.nonlinearity)(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        for size in self.mlp_sizes:
            x = nn.Dense(size)(x)
            x = act(x)
        return x


class ActorCritic(nn.Module):
    vision_encoder: nn.Module
    text_encoder: Optional[nn.Module] = None
    layer_width: int = 512
    action_dim: int = 6

    @nn.compact
    def __call__(self, image_input: jnp.ndarray, text_input: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        # Encode vision
        vision_feat = self.vision_encoder(image_input)

        # Encode text (use MLP if no custom encoder provided)
        text_feat = self.text_encoder(text_input) if self.text_encoder is not None else text_input

        # Combine features
        combined = jnp.concatenate([vision_feat, text_feat], axis=-1)

        # Actor
        actor_hidden = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined)
        actor_hidden = nn.relu(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic
        critic_hidden = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined)
        critic_hidden = nn.relu(critic_hidden)
        critic_value = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_hidden)

        return pi, jnp.squeeze(critic_value, axis=-1)


# Baselines --------------------------------------------------- 
class AC_IMG_conv_TXT_mlp_film(nn.Module):
    action_dim: int
    layer_width: int
    embed_dim: int = 64 

    def setup(self):
        self.dense1 = nn.Dense(self.embed_dim)
        self.dense2 = nn.Dense(self.embed_dim)
        self.norm = nn.LayerNorm()

    def process_text_embedding(self, text_embedding):
        x = self.dense1(text_embedding)
        x = nn.relu(x)
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.norm(x) 
        return x

    def compute_film_params(self, processed_embedding, num_channels):
        gamma = nn.Dense(num_channels)(processed_embedding)
        beta = nn.Dense(num_channels)(processed_embedding)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return gamma, beta
    
    @nn.compact
    def __call__(self, obs, text_embedding):

        processed_embedding = self.process_text_embedding(text_embedding)

        x_conv = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.relu(x_conv)

        gamma, beta = self.compute_film_params(processed_embedding, num_channels=32)
        x_film = gamma * x_conv + beta
        x = nn.max_pool(x_film + x_conv, window_shape=(3, 3), strides=(3, 3))

        x_conv = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.relu(x_conv)

        gamma, beta = self.compute_film_params(processed_embedding, num_channels=32)
        x_film = gamma * x_conv + beta
        x = nn.max_pool(x_film + x_conv, window_shape=(3, 3), strides=(3, 3))

        x_conv = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.relu(x_conv)

        gamma, beta = self.compute_film_params(processed_embedding, num_channels=32)
        x_film = gamma * x_conv + beta
        x = nn.max_pool(x_film + x_conv, window_shape=(3, 3), strides=(3, 3))

        image_embedding = x.reshape(x.shape[0], -1)

        combined_embedding = jnp.concatenate([image_embedding, processed_embedding], axis=-1)

        # Actor & residual
        actor_hidden = nn.Dense(self.layer_width)(combined_embedding)
        actor_hidden = nn.relu(actor_hidden)
        actor_residual = nn.Dense(self.layer_width)(actor_hidden)
        actor_hidden = nn.relu(actor_hidden + actor_residual)
        actor_logits = nn.Dense(self.action_dim)(actor_hidden)

        pi = distrax.Categorical(logits=actor_logits)

        # Critic & residual
        critic_hidden = nn.Dense(self.layer_width)(combined_embedding)
        critic_hidden = nn.relu(critic_hidden)
        critic_residual = nn.Dense(self.layer_width)(critic_hidden)
        critic_hidden = nn.relu(critic_hidden + critic_residual)
        critic_value = nn.Dense(1)(critic_hidden)

        return pi, jnp.squeeze(critic_value, axis=-1)

    
class ActorCriticConvWithBERT(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"
    bert_model_name: str = "bert-base-uncased"

    @nn.compact
    def __call__(self, obs, encoded_input):
        
        # Image processing (unchanged)
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        image_embedding = x.reshape(x.shape[0], -1)

        
        text_embedding = encoded_input

        # Combine image and text embeddings
        combined_embedding = jnp.concatenate([image_embedding, text_embedding], axis=-1)

        # Actor network
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Critic network
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        print("Done")
        return pi, jnp.squeeze(critic, axis=-1)


# Constructor
from typing import Optional, Sequence, Callable, Union
import flax.linen as nn
import distrax


def create_actor_critic(
    ac_type: str = "ac_model", #Alternativly: baseline, baseline_zoya, ac_film_model
    vision_type: str = "resnet_impala",  # 
    text_encoder_type: Optional[str] = None,  # "mlp", None (means default MLP), or custom
    text_mlp_sizes: Sequence[int] = (128, 128),
    nonlinearity: str = "relu",
    vision_mlp_sizes: Sequence[int] = (256,),
    layer_width: int = 512,
    action_dim: int = 22,
) -> ActorCritic:
    
    if ac_type == "baseline_zoya":
        return AC_IMG_conv_TXT_mlp_film(action_dim=action_dim, layer_width=layer_width)

    if ac_type == "baseline":
        return ActorCriticConvWithBERT(action_dim=action_dim, layer_width=layer_width)

    if vision_type == "resnet_impala":
        vision_encoder = ResnetImpalaEncoder(
            nonlinearity=nonlinearity, mlp_sizes=vision_mlp_sizes
        )
    else:
        raise ValueError(f"Unknown vision_type: {vision_type}")

    if text_encoder_type == "mlp":
        text_encoder = MLP(layer_sizes=list(text_mlp_sizes), activation_fn=get_nonlinearity(nonlinearity))
    else:
        text_encoder = None

    return ActorCritic(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        layer_width=layer_width,
        action_dim=action_dim,
    )
    
if __name__ == "__main__":
    model = create_actor_critic(ac_type="ac_model",
                                vision_type="resnet_impala",
                                text_encoder_type=None,
                                text_mlp_sizes=(128,128),
                                nonlinearity="relu",
                                vision_mlp_sizes=(256,),
                                layer_width=512,
                                action_dim=22)