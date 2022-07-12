#!/bin/bash
#SBATCH -p g48
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -o /home/cbotos/github/qbes/imagenet-exp/SLURM_OUT/%A_%a.%x.out # STDOUT
#SBATCH -e /home/cbotos/github/qbes/imagenet-exp/SLURM_OUT/%A_%a.%x.err # STDERR
eval "$(conda shell.bash hook)"
source /home/cbotos/.sbatch-bashrc

cd /home/cbotos/github/qbes/imagenet-exp/
conda activate ffcv

echo "K=$1"
echo "BLOCK_SIZE=$2"

export CHOOSE=$1
export WEIGHTS=/export/work/cbotos/qbes/imagenet-logs/7e765429-20ec-4a59-8dd2-a51d933f8537/final_weights.pt

BLOCK_SIZE=$2

ID_FROM=$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))
ID_TO=$(((SLURM_ARRAY_TASK_ID+1) * BLOCK_SIZE))

echo "ID_FROM=$ID_FROM"
echo "ID_TO=$ID_TO"

python  test_imagenet.py --config rn50_configs/rn50_qbes_eval_train.yaml \
    --model.weights $WEIGHTS \
    --validation.proportion=0.1 \
    --logging.folder=$WORK/qbes/imagenet-test-logs/cache/train/0.1 \
    --qbes.config_file qbes_configs/16nCr$CHOOSE.json \
    --qbes.id_from $ID_FROM --qbes.id_to $ID_TO \
    --training.distributed 0 --dist.world_size 1