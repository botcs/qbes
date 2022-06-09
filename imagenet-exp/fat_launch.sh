sbatch --qos=high --partition=g24 --array=0-1 eval-cluster/training-set/choose3.sh
sbatch --qos=high --partition=g48 --array=0-3 eval-cluster/training-set/choose4.sh
sbatch --qos=normal --partition=g48 --array=0-3 eval-cluster/training-set/choose5.sh
sbatch --qos=normal --partition=g24 --array=4-8 eval-cluster/training-set/choose5.sh
sbatch --qos=normal --partition=g24 --array=0-5 eval-cluster/training-set/choose6.sh
sbatch --qos=normal --partition=g48 --array=6-11 eval-cluster/training-set/choose6.sh
sbatch --qos=low --partition=g24 --array=0-11 eval-cluster/training-set/choose7.sh
sbatch --qos=low --partition=g48 --array=12-22 eval-cluster/training-set/choose7.sh