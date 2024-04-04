#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ppo
#SBATCH --array=1-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=shivakanth.sujit@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# source ~/ENV/bin/activate

env=${1:-"room-multi-passage"}
sparse_reward=${2:-False}
timelimit=200

dynamics_model=planet
# dynamics_model=planet_pclast

logdir="logs"

num_episodes=2500

# ! ONLY FOR EVAL
test_args="eval_only=True"

# logdir="${logdir}/testing"
SLURM_ARRAY_TASK_ID=0
# num_episodes=2
# env_args="${env_args}"
# timelimit=20
#SBATCH --account=def-bengioy
#SBATCH --account=rrg-bengioy-ad

test_args="${test_args} experiment="cem_diff" overrides.cem_num_iters=2 overrides.cem_population_size=100"

eval_eps=100
env_args="${env_args} eval_eps=${eval_eps} env_name=${env} sparse_reward=${sparse_reward} overrides.trial_length=${timelimit}"

seed=${SLURM_ARRAY_TASK_ID}

python main.py root_dir=${logdir} seed=${seed} ${env_args} dynamics_model=${dynamics_model} algorithm.num_episodes=${num_episodes} algorithm.test_frequency=50 ${test_args}