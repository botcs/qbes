#!/bin/bash
sbatch --array=0-0 --job-name="K0-val" eval-cluster/validation-set-full/chooseK.sh 0 100
sbatch --array=0-0 --job-name="K1-val" eval-cluster/validation-set-full/chooseK.sh 1 100
sbatch --array=0-1 --job-name="K2-val" eval-cluster/validation-set-full/chooseK.sh 2 100
sbatch --array=0-5 --job-name="K3-val" eval-cluster/validation-set-full/chooseK.sh 3 100
sbatch --array=0-18 --job-name="K4-val" eval-cluster/validation-set-full/chooseK.sh 4 100
sbatch --array=0-43 --job-name="K5-val" eval-cluster/validation-set-full/chooseK.sh 5 100
sbatch --array=0-80 --job-name="K6-val" eval-cluster/validation-set-full/chooseK.sh 6 100
sbatch --array=0-114 --job-name="K7-val" eval-cluster/validation-set-full/chooseK.sh 7 100
sbatch --array=0-128 --job-name="K8-val" eval-cluster/validation-set-full/chooseK.sh 8 100
sbatch --array=0-114 --job-name="K9-val" eval-cluster/validation-set-full/chooseK.sh 9 100
sbatch --array=0-80 --job-name="K10-val" eval-cluster/validation-set-full/chooseK.sh 10 100
sbatch --array=0-43 --job-name="K11-val" eval-cluster/validation-set-full/chooseK.sh 11 100
sbatch --array=0-18 --job-name="K12-val" eval-cluster/validation-set-full/chooseK.sh 12 100
sbatch --array=0-5 --job-name="K13-val" eval-cluster/validation-set-full/chooseK.sh 13 100
sbatch --array=0-1 --job-name="K14-val" eval-cluster/validation-set-full/chooseK.sh 14 100
sbatch --array=0-0 --job-name="K15-val" eval-cluster/validation-set-full/chooseK.sh 15 100
sbatch --array=0-0 --job-name="K16-val" eval-cluster/validation-set-full/chooseK.sh 16 100
