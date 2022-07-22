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
echo "ID_OFFSET=$3 (optional)"

export CHOOSE=$1

BLOCK_SIZE=$2

ID_FROM=$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE + ID_OFFSET))
ID_TO=$(((SLURM_ARRAY_TASK_ID+1) * BLOCK_SIZE + ID_OFFSET))

echo "ID_FROM=$ID_FROM"
echo "ID_TO=$ID_TO"

python test_imagenet.py --config swin_configs/swin_qbes_eval_train.yaml \
    --model.weights "" --model.arch "swin_b" \
    --qbes.config_file qbes_configs/24nCr$CHOOSE-10k.json \
    --validation.proportion=0.1 \
    --logging.folder=$WORK/qbes/imagenet-test-logs/swin/cache/train/0.1-10kCAP/ \
    --qbes.id_from $ID_FROM --qbes.id_to $ID_TO \
    --training.distributed 0 --dist.world_size 1