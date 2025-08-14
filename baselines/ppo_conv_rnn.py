import argparse
import logging
import os
import sys
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from analysis.inference_rnn import Experiment, ExperimentArgs
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.encoders.craftext_distilbert_model_encoder import make_encoder
from craftext.environment.scenarious.manager import create_scenarios_with_dataset
from flax.training import (
    orbax_utils,
)
from flax.training.train_state import TrainState
from clu import metric_writers

from logz.batch_logging import batch_log_tb, create_log_dict
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)
from rnn_network import ActorCriticTextVisualRNN, ScannedRNN
from wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@flax.struct.dataclass
class TransitionScheme:
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: jax.Array
    instruction: jax.Array


def make_train(config, network_params, writer=None):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_PER_ITERATION"] = (
        config["ITERATION_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    if config["USE_PLANS"]:
        encoder = make_encoder(n_splits=1)
        scenarious_loader = create_scenarios_with_dataset(True)
        env = InstructionWrapper(
            env,
            config["CRAFTEXT_SETTINGS"],
            encode_model_class=encoder,
            scenario_handler_class=scenarious_loader,
        )
    else:
        encoder = make_encoder(n_splits=config["EXPAND_EMB"])
        env = InstructionWrapper(
            env, config["CRAFTEXT_SETTINGS"], encode_model_class=encoder
        )
    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticTextVisualRNN(
            env.action_space(env_params).n, config=config
        )
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )

        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        encoded = env.encoded_instruction  # (768,)
        encoded = jnp.expand_dims(encoded, axis=0)  # (1, 768)
        encoded_input_tiled = jnp.tile(encoded, (1, config["NUM_ENVS"], 1))
        network_params_alt = network.init(
            _rng, init_hstate, init_x, encoded_input_tiled
        )
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        if network_params is None:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params_alt,
                tx=tx,
            )
        else:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                print("-->", env_state.env_state.instruction.shape)
                print("->", last_obs[np.newaxis, :].shape)
                print("->", ac_in[0].shape)
                hstate, pi, value = network.apply(
                    train_state.params,
                    hstate,
                    ac_in,
                    env_state.env_state.instruction[np.newaxis, :],
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = TransitionScheme(
                    last_done,
                    action,
                    value,
                    reward,
                    log_prob,
                    last_obs,
                    info,
                    instruction=env_state.env_state.instruction,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(
                train_state.params,
                hstate,
                ac_in,
                env_state.env_state.instruction[np.newaxis, :],
            )
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate[0],
                            (traj_batch.obs, traj_batch.done),
                            traj_batch.instruction,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_TB"] and writer is not None:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log_tb(update_step, to_log, config, writer)

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            config["UPDATE_STEP"],
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES_PER_ITERATION"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_ppo(config):
    # Convert config keys to uppercase for consistency
    config = {k.upper(): v for k, v in config.__dict__.items()}

    config["PATH_TO_CHECKPOINT"] = "None"

    writer = None

    if config["USE_TB"]:
        if config["USE_DATE"]: 
            run_name = f"{config['WANDB_RUN']}_{time.strftime('%Y%m%d-%H%M%S')}"
        else:
            run_name = f"{config['WANDB_RUN']}"

        log_dir = os.path.join("runs", run_name)
        writer = metric_writers.create_default_writer(logdir=log_dir) 
        logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    checkpoint_dir = os.path.abspath(
        os.path.join("./experiments", run_name, "checkpoints")
    )
    os.makedirs(checkpoint_dir, exist_ok=True)  # Важно создать папку

    orbax_checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,  # Используем новый, чистый путь
        orbax_checkpointer,
        CheckpointManagerOptions(max_to_keep=config["NUM_RESTARTS"], create=True),
    )

    config["PATH_TO_CHECKPOINT"] = "None"

    # Initialize random keys
    rng = jax.random.PRNGKey(config["SEED"])

    # Define the number of restarts
    num_restarts = config["NUM_RESTARTS"]  # Hyperparameter for the number of restarts
    config["ITERATION_STEPS"] = config["TOTAL_TIMESTEPS"] // num_restarts

    config["NUM_UPDATES"] = (
        config["ITERATION_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["UPDATE_STEP"] = config["START_RESTART"] * config["NUM_UPDATES"]

    for restart in range(config["START_RESTART"], num_restarts):
        config["CURRENT_RESTART"] = restart + 1
        print(f"Starting training iteration {config['CURRENT_RESTART']}/{num_restarts}")

        with jax.disable_jit():
            if config["CURRENT_RESTART"] > 1:
                train_state = checkpoint_manager.restore(
                    int(config["ITERATION_STEPS"] * (config["CURRENT_RESTART"] - 1))
                )
                network_params = train_state["runner_state"][0]["params"]
                print("Weights successfully loaded from checkpoint.")
            else:
                print("No valid checkpoint found, using default initialization.")
                network_params = None  # Initialize or handle default weights

        # Split RNG for this training iteration
        rng, current_rng = jax.random.split(rng)

        # Prepare the training function
        train_jit = jax.jit(make_train(config, network_params, writer))

        # Run the training
        t0 = time.time()
        train_state = train_jit(current_rng)
        t1 = time.time()

        # Print performance metrics
        logger.info(f"Iteration {config['CURRENT_RESTART']} completed.")
        logger.info(f"Time to run experiment: {t1 - t0}")
        logger.info(
            f"steps: from {config['ITERATION_STEPS'] * config['CURRENT_RESTART'] - 1} to {config['ITERATION_STEPS'] * config['CURRENT_RESTART']}"
        )
        logger.info(f"SPS: {config['ITERATION_STEPS'] / (t1 - t0)}")

        config["UPDATE_STEP"] = train_state["runner_state"][-1]

        # Save the current train state
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            config["ITERATION_STEPS"] * (config["CURRENT_RESTART"]),
            train_state,
            save_kwargs={"save_args": save_args},
        )

        # INFERENCE
        common_args = {
            "num_envs": config["NUM_ENVS"],
            "experiment_name": log_dir,
            "ratio": min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
            "checkpoint_num": config["PATH_TO_CHECKPOINT"],
            "env_name": config["ENV_NAME"],
            "max_grad_norm": config["MAX_GRAD_NORM"],
            "lr": config["LR"],
            "layer_size": config["LAYER_SIZE"],
            "total_timesteps": config["ITERATION_STEPS"] * (config["CURRENT_RESTART"]),
            "path": checkpoint_dir,
            "view": False,
            "use_plans": config["USE_PLANS"],
            "inference_step": config["INFERENCE_STEP"],
            "expand_emb": config["EXPAND_EMB"],
        }
        
        def log_inference_to_tensorboard(prefix, data, step):
            if writer:
                for task_name, value in data.items():
                    writer.write_scalars(step, {f"{prefix}/{task_name}": value})
                logger.info(f"Logged inference '{prefix}' metrics at step {step}")

        # INFERENCE ON TRAIN
        train_args = ExperimentArgs(
            **common_args, craftext_settings=config["CRAFTEXT_SETTINGS"]
        )
        train_experiment = Experiment(train_args)
        inference, mean_by_tasks = train_experiment.run()

        if config["USE_TB"]:
            log_inference_to_tensorboard("train", mean_by_tasks, config["CURRENT_RESTART"])

        # INFERENCE ON TEST
        test_args = ExperimentArgs(
            **common_args,
            craftext_settings=config["CRAFTEXT_SETTINGS"].replace(
                "_train", "_test_other_params"
            ),
        )
        test_experiment = Experiment(test_args)
        inference, mean_by_tasks = test_experiment.run()

        if config["USE_TB"]:
            log_inference_to_tensorboard("test", mean_by_tasks, config["CURRENT_RESTART"])

        # INFERENCE ON PARAPHRASED
        test_paraphrases_args = ExperimentArgs(
            **common_args,
            craftext_settings=config["CRAFTEXT_SETTINGS"].replace(
                "_train", "_test_paraphrases"
            ),
        )
        test_paraphrases_experiment = Experiment(test_paraphrases_args)
        inference, mean_by_tasks = test_paraphrases_experiment.run()

        if config["USE_TB"]:
            log_inference_to_tensorboard("test_paraphrases", mean_by_tasks, config["CURRENT_RESTART"])

    if writer:
        writer.flush()
    print("All training iterations completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--expand_emb", type=int, default=1)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--use_plans", type=bool, default=False)
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=250000000
    )
    parser.add_argument("--num_restarts", type=lambda x: int(float(x)), default=5)
    parser.add_argument("--start_restart", type=int, default=0)
    parser.add_argument("--inference_step", type=lambda x: int(float(x)), default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use_tb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_run", type=str)
    parser.add_argument(
        "--use_date", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
