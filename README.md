# 🌟 Baselines for the CrafText Benchmark

## Advancing Instruction Following in Complex Multimodal Open-Ended Worlds

**CrafText** is a goal-conditioned extension of the [Craftax environment](https://github.com/MichaelTMatthews/Craftax), specifically developed as a benchmark for multimodal reinforcement learning.
It challenges agents to follow natural language instructions grounded in rich, visual environments inspired by Minecraft. Agents must combine visual perception** and language understanding to execute complex action sequences in dynamic, open-ended worlds.

🔗 **Environment Repository:**
[https://github.com/AIRI-Institute/CrafText](https://github.com/AIRI-Institute/CrafText)



## ✨ Baseline Overview

* **🟡 PPO-T**: A baseline based on the PPO algorithm where the agent encodes instructions using **BERT embeddings** and processes them jointly with visual observations.
* **🟢 PPO-T+**: An enhanced version of PPO-T that leverages **high-level plans** to improve instruction following.
* **🔵 FiLM**: A model using **feature-wise modulation (FiLM)** to dynamically condition policy behavior on language instructions.
* **🟣 LLM Baseline**: Zero-shot evaluation using **GPT-4**, **LLaMA**, **Qwen**, and other large language models via API access and models from Hugging Face.

---

## 🚀 Running Experiments

Below are the available baselines with example commands to launch training runs.


### 🟡 PPO-T Baseline

```bash
python3 baselines/ppo_conv_rnn.py \
    --craftext_settings easy_train \
    --env_name "Craftax-Classic-Pixels-v1" \
    --num_envs 1024 \
    --total_timesteps 6000000000 \
```

### 🟢 PPO-T+ Baseline

```bash
python3 baselines/ppo_conv_rnn.py \
    --craftext_settings easy_train \
    --env_name "Craftax-Classic-Pixels-v1" \
    --num_envs 1024 \
    --total_timesteps 6000000000 \
    --use_plans True \
```



### 🔵 FiLM Baseline

```bash
python3 baselines/ppo_filmed.py \
    --craftext_settings easy_train \
    --env_name "Craftax-Classic-Pixels-v1" \
    --num_envs 1024 \
    --total_timesteps 6000000000 \
```



### 🟣 LLM (Zero-Shot) Baseline

```bash
python llm_baselines/run.py \
    --model_source huggingface \
    --model_name Qwen/Qwen2.5-0.5B \
    --output_file qwen_results.txt
```

