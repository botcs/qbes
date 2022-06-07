#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 288
eval "$(conda shell.bash hook)"
source /home/cbotos/.sbatch-bashrc

cd /home/cbotos/github/ffcv-imagenet/
conda activate ffcv
export IMAGENET_DIR=/export/share/datasets/ILSVRC2012/
export WRITE_DIR=$WORK/imagenet/

./write_imagenet.sh 500 0.50 90
