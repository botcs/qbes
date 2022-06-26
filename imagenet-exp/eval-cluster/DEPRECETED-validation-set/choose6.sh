#!/bin/bash
#SBATCH -p g48
#SBATCH -c 12
#SBATCH --gres=gpu:8
#SBATCH --qos=low
#SBATCH -o /home/cbotos/github/qbes/imagenet-exp/SLURM_OUT/%A_%a.%x.out # STDOUT
#SBATCH -e /home/cbotos/github/qbes/imagenet-exp/SLURM_OUT/%A_%a.%x.err # STDERR
eval "$(conda shell.bash hook)"
source /home/cbotos/.sbatch-bashrc

cd /home/cbotos/github/qbes/imagenet-exp/
conda activate ffcv

export CHOOSE=6
export WEIGHTS=/export/work/cbotos/qbes/imagenet-logs/7e765429-20ec-4a59-8dd2-a51d933f8537/final_weights.pt

BLOCK_SIZE=700
ID_FROM=$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))
ID_TO=$(((SLURM_ARRAY_TASK_ID+1) * BLOCK_SIZE))

python  test_imagenet.py --config rn50_configs/rn50_test.yaml \
    --model.weights $WEIGHTS \
    --logging.folder=$WORK/qbes/imagenet-test-logs/ \
    --qbes.config_file qbes_configs/16nCr$CHOOSE.json \
    --qbes.id_from $ID_FROM --qbes.id_to $ID_TO \
    --dist.world_size 8