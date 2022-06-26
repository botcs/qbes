#!/bin/bash
sbatch --array=0-0 --job-name="K0-val" eval-cluster/validation-set-full/chooseK.sh 0 25
sbatch --array=0-0 --job-name="K1-val" eval-cluster/validation-set-full/chooseK.sh 1 25
sbatch --array=0-4 --job-name="K2-val" eval-cluster/validation-set-full/chooseK.sh 2 25
sbatch --array=0-22 --job-name="K3-val" eval-cluster/validation-set-full/chooseK.sh 3 25
sbatch --array=0-72 --job-name="K4-val" eval-cluster/validation-set-full/chooseK.sh 4 25
sbatch --array=0-174 --job-name="K5-val" eval-cluster/validation-set-full/chooseK.sh 5 25
sbatch --array=0-320 --job-name="K6-val" eval-cluster/validation-set-full/chooseK.sh 6 25
sbatch --array=0-457 --job-name="K7-val" eval-cluster/validation-set-full/chooseK.sh 7 25
sbatch --array=0-514 --job-name="K8-val" eval-cluster/validation-set-full/chooseK.sh 8 25
sbatch --array=0-457 --job-name="K9-val" eval-cluster/validation-set-full/chooseK.sh 9 25
sbatch --array=0-320 --job-name="K10-val" eval-cluster/validation-set-full/chooseK.sh 10 25
sbatch --array=0-174 --job-name="K11-val" eval-cluster/validation-set-full/chooseK.sh 11 25
sbatch --array=0-72 --job-name="K12-val" eval-cluster/validation-set-full/chooseK.sh 12 25
sbatch --array=0-22 --job-name="K13-val" eval-cluster/validation-set-full/chooseK.sh 13 25
sbatch --array=0-4 --job-name="K14-val" eval-cluster/validation-set-full/chooseK.sh 14 25
sbatch --array=0-0 --job-name="K15-val" eval-cluster/validation-set-full/chooseK.sh 15 25
sbatch --array=0-0 --job-name="K16-val" eval-cluster/validation-set-full/chooseK.sh 16 25