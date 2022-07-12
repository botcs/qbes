#!/bin/bash
sbatch --array=0-0 --job-name="K0-train" eval-cluster/training-set-0.1/chooseK.sh 0 50
sbatch --array=0-0 --job-name="K1-train" eval-cluster/training-set-0.1/chooseK.sh 1 50
sbatch --array=0-2 --job-name="K2-train" eval-cluster/training-set-0.1/chooseK.sh 2 50
sbatch --array=0-11 --job-name="K3-train" eval-cluster/training-set-0.1/chooseK.sh 3 50
sbatch --array=0-36 --job-name="K4-train" eval-cluster/training-set-0.1/chooseK.sh 4 50
sbatch --array=0-87 --job-name="K5-train" eval-cluster/training-set-0.1/chooseK.sh 5 50
sbatch --array=0-160 --job-name="K6-train" eval-cluster/training-set-0.1/chooseK.sh 6 50
sbatch --array=0-228 --job-name="K7-train" eval-cluster/training-set-0.1/chooseK.sh 7 50
sbatch --array=0-257 --job-name="K8-train" eval-cluster/training-set-0.1/chooseK.sh 8 50
sbatch --array=0-228 --job-name="K9-train" eval-cluster/training-set-0.1/chooseK.sh 9 50
sbatch --array=0-160 --job-name="K10-train" eval-cluster/training-set-0.1/chooseK.sh 10 50
sbatch --array=0-87 --job-name="K11-train" eval-cluster/training-set-0.1/chooseK.sh 11 50
sbatch --array=0-36 --job-name="K12-train" eval-cluster/training-set-0.1/chooseK.sh 12 50
sbatch --array=0-11 --job-name="K13-train" eval-cluster/training-set-0.1/chooseK.sh 13 50
sbatch --array=0-2 --job-name="K14-train" eval-cluster/training-set-0.1/chooseK.sh 14 50
sbatch --array=0-0 --job-name="K15-train" eval-cluster/training-set-0.1/chooseK.sh 15 50
sbatch --array=0-0 --job-name="K16-train" eval-cluster/training-set-0.1/chooseK.sh 16 50