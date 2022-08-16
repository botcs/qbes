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

# TEMPLATE END
################################################
# AUTOMATICALLY GENERATED ON:
# 2022-08-16 09:59:20.833953
################################################
VALS=(regression classification)
ARG=mlp.prediction_type
VAL=${VALS[$SLURM_ARRAY_TASK_ID]}
python train_mlp.py --config swin_configs/swin_K_mlp.yaml --$ARG $VAL --logging.folder logs/swin/mlp/$ARG"__"$VAL/ 