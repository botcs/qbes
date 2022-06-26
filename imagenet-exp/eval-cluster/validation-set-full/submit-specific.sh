#!/bin/bash
# These are done already
# --array=0-0 eval-cluster/training-set-0.1/chooseK.sh 0
sbatch --array=0-0 eval-cluster/training-set-0.1/chooseK.sh 1
# sbatch --array=0-1 eval-cluster/training-set-0.1/chooseK.sh 2
# sbatch --array=0-5 eval-cluster/training-set-0.1/chooseK.sh 3
# sbatch --array=0-18 eval-cluster/training-set-0.1/chooseK.sh 4
# sbatch --array=0-43 eval-cluster/training-set-0.1/chooseK.sh 5
# sbatch --array=0-80 eval-cluster/training-set-0.1/chooseK.sh 6
# sbatch --array=0-114 eval-cluster/training-set-0.1/chooseK.sh 7
sbatch --array=0-128 eval-cluster/training-set-0.1/chooseK.sh 8
sbatch --array=0-114 eval-cluster/training-set-0.1/chooseK.sh 9
sbatch --array=0-80 eval-cluster/training-set-0.1/chooseK.sh 10
sbatch --array=0-43 eval-cluster/training-set-0.1/chooseK.sh 11
sbatch --array=0-18 eval-cluster/training-set-0.1/chooseK.sh 12
sbatch --array=0-5 eval-cluster/training-set-0.1/chooseK.sh 13
sbatch --array=0-1 eval-cluster/training-set-0.1/chooseK.sh 14
sbatch --array=0-0 eval-cluster/training-set-0.1/chooseK.sh 15
sbatch --array=0-0 eval-cluster/training-set-0.1/chooseK.sh 16