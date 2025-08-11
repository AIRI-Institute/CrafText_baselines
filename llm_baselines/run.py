import os

import time
import argparse
import jax
import torch
from tqdm import tqdm

from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from llm_agent import HuggingfaceAgent
from api_agent import ApiAgent
from utils import render_craftax_text



def parse_args():
    parser = argparse.ArgumentParser(description="Run agent in Craftax environment")
    parser.add_argument("--output_file", type=str, default="results.txt", help="Output file for success rates")
    parser.add_argument("--model_source", type=str, choices=["huggingface", "api"], default="huggingface", help="Model source: huggingface or api")
    parser.add_argument("--model_name", type=str, default="", help="Model name for the agent")
    parser.add_argument("--craftext_settings", type=str, default="easy_train", help="Craftext settings")
    return parser.parse_args()

def main():
    args = parse_args()

    env_name = "Craftax-Classic-Symbolic-v1"
    use_optimistic_resets = False
    env = make_craftax_env_from_name(env_name, not use_optimistic_resets)
    env_params = env.default_params
    env = InstructionWrapper(env, args.craftext_settings)

    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(1)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, 3)

    instruction_list = env.scenario_handler.scenario_data.instructions_list

    if args.model_source == "huggingface":
        default_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        agent = HuggingfaceAgent(model_name=args.model_name or default_model)
    else:
        agent = ApiAgent(api_name=args.model_name or "openai")

    SR = []

    # for i in tqdm(range(len(instruction_list))):
    for i in tqdm(range(2)):
        obsv, env_state = env.reset(_rng, env_params, instruction_idx=i)
        agent.reset_state()

        done = False
        steps_limit = 250
        while not done:
            instruction_indx_run = env_state.idx.item()
            instruction = instruction_list[instruction_indx_run]
            text_observation = render_craftax_text(env_state.env_state)
            text_instruction = instruction
            start = time.time()
            action = agent.act(text_observation=text_observation, instruction=text_instruction)
            obs, env_state, reward, done, info = step_fn(rngs[2], env_state, action, env_params)
            end = time.time()
            print(f"Execution time: {end - start:.6f} seconds")
            if info['SR'] > 0:
                info['SR'] = 1
                break
            steps_limit -= 1
            if steps_limit <= 0:
                break
        SR.append(float(info['SR']))
        with open(args.output_file, "w") as f:
            f.write(str(SR))

if __name__ == '__main__':
    main()
