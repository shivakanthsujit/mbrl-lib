#!/bin/bash

envs="polygon-obs room-multi-passage room-spiral"
# envs="polygon-obs room-spiral"
# sparse_rewards="True False"
# envs="polygon-obs"
sparse_rewards="True False"
sparse_rewards="True"
# sparse_rewards="False"
# opts="FF TF FT TT"


for env in $envs
do
    for sparse_reward in $sparse_rewards
    do      
            run_cmd="exp.sh ${env} ${sparse_reward}"
            sbatch_cmd="sbatch ${run_cmd}"
            cmd="$sbatch_cmd"
            # cmd="bash $run_cmd"
            echo -e "${cmd}"
            ${cmd}
            sleep 1
    done
done