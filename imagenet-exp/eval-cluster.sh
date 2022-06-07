#!/bin/bash
#SBATCH -p g24
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
eval "$(conda shell.bash hook)"
source /home/cbotos/.sbatch-bashrc

cd /home/cbotos/qbes/imagenet-exp
conda activate ffcv

export CUDA_VISIBLE_DEVICES=0
export WEIGHTS=/export/work/cbotos/qbes/imagenet-logs/7e765429-20ec-4a59-8dd2-a51d933f8537/final_weights.pt

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
BLOCK_SIZE=500
ID_FROM=$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))
ID_TO=$(((SLURM_ARRAY_TASK_ID+1) * BLOCK_SIZE))

python  test_imagenet.py --config rn50_configs/rn50_test.yaml \
    --model.weights $WEIGHTS \
    --logging.folder=$WORK/qbes/imagenet-test-logs/ \
    --qbes.config_file qbes_configs/16nCr7.json \
    --qbes.id_from $ID_FROM --qbes.id_to $ID_TO