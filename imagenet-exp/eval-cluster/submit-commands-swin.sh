#!/bin/bash
sbatch --array=0-0 --job-name="K0-train" eval-cluster/training-set-0.1/chooseK-swin.sh 0 20
sbatch --array=0-1 --job-name="K1-train" eval-cluster/training-set-0.1/chooseK-swin.sh 1 20
sbatch --array=0-13 --job-name="K2-train" eval-cluster/training-set-0.1/chooseK-swin.sh 2 20
sbatch --array=0-101 --job-name="K3-train" eval-cluster/training-set-0.1/chooseK-swin.sh 3 20
sbatch --array=0-101 --job-name="K21-train" eval-cluster/training-set-0.1/chooseK-swin.sh 21 20
sbatch --array=0-13 --job-name="K22-train" eval-cluster/training-set-0.1/chooseK-swin.sh 22 20
sbatch --array=0-1 --job-name="K23-train" eval-cluster/training-set-0.1/chooseK-swin.sh 23 20
sbatch --array=0-0 --job-name="K24-train" eval-cluster/training-set-0.1/chooseK-swin.sh 24 20

sbatch --array=0-0 --job-name="K0-val" eval-cluster/validation-set-full/chooseK-swin.sh 0 20
sbatch --array=0-1 --job-name="K1-val" eval-cluster/validation-set-full/chooseK-swin.sh 1 20
sbatch --array=0-13 --job-name="K2-val" eval-cluster/validation-set-full/chooseK-swin.sh 2 20
sbatch --array=0-101 --job-name="K3-val" eval-cluster/validation-set-full/chooseK-swin.sh 3 20
sbatch --array=0-101 --job-name="K21-val" eval-cluster/validation-set-full/chooseK-swin.sh 21 20
sbatch --array=0-13 --job-name="K22-val" eval-cluster/validation-set-full/chooseK-swin.sh 22 20
sbatch --array=0-1 --job-name="K23-val" eval-cluster/validation-set-full/chooseK-swin.sh 23 20
sbatch --array=0-0 --job-name="K24-val" eval-cluster/validation-set-full/chooseK-swin.sh 24 20

# Thicc runs
sbatch --array=0-531 --job-name="K4-train" eval-cluster/training-set-0.1/chooseK-swin.sh 4 20
sbatch --array=0-531 --job-name="K20-train" eval-cluster/training-set-0.1/chooseK-swin.sh 20 20

sbatch --array=0-531 --job-name="K4-val" eval-cluster/validation-set-full/chooseK-swin.sh 4 20
sbatch --array=0-531 --job-name="K20-val" eval-cluster/validation-set-full/chooseK-swin.sh 20 20