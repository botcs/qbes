#!/bin/bash
#SBATCH -p g48
#SBATCH -c 12
#SBATCH --gres=gpu:2
#SBATCH --qos=normal
#SBATCH -o /home/cbotos/github/qbes/imagenet-exp/JUPYTER_OUT/OUT # STDOUT
#SBATCH -e /home/cbotos/github/qbes/imagenet-exp/JUPYTER_OUT/ERR # STDERR
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=botos.official@gmail.com
eval "$(conda shell.bash hook)"
source /home/cbotos/.sbatch-bashrc

cd /home/cbotos/github/qbes/imagenet-exp/

conda activate qbes-vis
jupyter-notebook --ip=0.0.0.0