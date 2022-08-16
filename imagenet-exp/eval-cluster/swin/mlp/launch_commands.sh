sbatch --array=0-2 --job-name=lr eval-cluster/swin/mlp//lr.lr__array.sh
sbatch --array=0-1 --job-name=lr_schedule_type eval-cluster/swin/mlp//lr.lr_schedule_type__array.sh
sbatch --array=0-2 --job-name=batch_size eval-cluster/swin/mlp//training.batch_size__array.sh
sbatch --array=0-1 --job-name=trunk_layer eval-cluster/swin/mlp//mlp.trunk_layer__array.sh
sbatch --array=0-1 --job-name=prediction_type eval-cluster/swin/mlp//mlp.prediction_type__array.sh
sbatch --array=0-1 --job-name=balance_weight eval-cluster/swin/mlp//mlp.balance_weight__array.sh
