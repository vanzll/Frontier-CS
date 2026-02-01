#!/bin/bash
# Run verl-agent training with custom Solution.
#
# This script matches verl-agent's original run_alfworld.sh configuration,
# but uses algorithm.adv_estimator=custom to invoke the user's Solution.

set -x
ENGINE=${ENGINE:-vllm}
# Use FLASH_ATTN backend for vLLM 0.11.0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Ensure Ray logs are visible
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_AGENT_DIR="${SCRIPT_DIR}/verl-agent"

# Solution path from environment variable
SOLUTION_PATH="${SOLUTION_PATH:-/workspace/solution.py}"

# Set ALFWorld data path
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
echo "ALFWORLD_DATA=${ALFWORLD_DATA}"

# Training configuration (from verl-agent standard run_alfworld.sh)
num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8

echo "=== Running verl-agent Training ==="
echo "Solution: ${SOLUTION_PATH}"
echo "Engine: ${ENGINE}"
echo ""

# Prepare data
cd "${VERL_AGENT_DIR}"
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Run training with custom solution
# Configuration matches verl-agent/examples/grpo_trainer/run_alfworld.sh exactly,
# except algorithm.adv_estimator=custom for user-defined advantage calculation.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=custom \
    +algorithm.custom.solution_path="${SOLUTION_PATH}" \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='agentic_rl_discovery' \
    trainer.experiment_name='custom_solution' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True

echo "=== Training Complete ==="
