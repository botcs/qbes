# sbatch --array=0-0 --job-name="K0-val" eval-cluster/validation-set-full/chooseK-swin.sh 0 20
# sbatch --array=0-1 --job-name="K1-val" eval-cluster/validation-set-full/chooseK-swin.sh 1 20
# sbatch --array=0-13 --job-name="K2-val" eval-cluster/validation-set-full/chooseK-swin.sh 2 20
# sbatch --array=0-101 --job-name="K3-val" eval-cluster/validation-set-full/chooseK-swin.sh 3 20
sbatch --array=0-531 --job-name="K4-val" eval-cluster/validation-set-full/chooseK-swin.sh 4 20
sbatch --array=0-531 --job-name="K20-val" eval-cluster/validation-set-full/chooseK-swin.sh 20 20
sbatch --array=0-101 --job-name="K21-val" eval-cluster/validation-set-full/chooseK-swin.sh 21 20
sbatch --array=0-13 --job-name="K22-val" eval-cluster/validation-set-full/chooseK-swin.sh 22 20
sbatch --array=0-1 --job-name="K23-val" eval-cluster/validation-set-full/chooseK-swin.sh 23 20
sbatch --array=0-0 --job-name="K24-val" eval-cluster/validation-set-full/chooseK-swin.sh 24 20
# CLAMPED @ 10k
sbatch --array=0-500 --job-name="K5-val" eval-cluster/validation-set-full/chooseK-swin.sh 5 20
sbatch --array=0-500 --job-name="K6-val" eval-cluster/validation-set-full/chooseK-swin.sh 6 20
sbatch --array=0-500 --job-name="K7-val" eval-cluster/validation-set-full/chooseK-swin.sh 7 20
sbatch --array=0-500 --job-name="K8-val" eval-cluster/validation-set-full/chooseK-swin.sh 8 20
sbatch --array=0-500 --job-name="K9-val" eval-cluster/validation-set-full/chooseK-swin.sh 9 20
sbatch --array=0-500 --job-name="K10-val" eval-cluster/validation-set-full/chooseK-swin.sh 10 20
sbatch --array=0-500 --job-name="K11-val" eval-cluster/validation-set-full/chooseK-swin.sh 11 20
sbatch --array=0-500 --job-name="K12-val" eval-cluster/validation-set-full/chooseK-swin.sh 12 20
sbatch --array=0-500 --job-name="K13-val" eval-cluster/validation-set-full/chooseK-swin.sh 13 20
sbatch --array=0-500 --job-name="K14-val" eval-cluster/validation-set-full/chooseK-swin.sh 14 20
sbatch --array=0-500 --job-name="K15-val" eval-cluster/validation-set-full/chooseK-swin.sh 15 20
sbatch --array=0-500 --job-name="K16-val" eval-cluster/validation-set-full/chooseK-swin.sh 16 20
sbatch --array=0-500 --job-name="K17-val" eval-cluster/validation-set-full/chooseK-swin.sh 17 20
sbatch --array=0-500 --job-name="K18-val" eval-cluster/validation-set-full/chooseK-swin.sh 18 20
sbatch --array=0-500 --job-name="K19-val" eval-cluster/validation-set-full/chooseK-swin.sh 19 20