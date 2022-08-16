import datetime

SCRIPT_DIR = "eval-cluster/swin/mlp/"

SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH -p g48
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -o /home/cbotos/github/qbes/imagenet-exp/SLURM_OUT/%A_%a.%x.out # STDOUT
#SBATCH -e /home/cbotos/github/qbes/imagenet-exp/SLURM_OUT/%A_%a.%x.err # STDERR
eval "$(conda shell.bash hook)"
source /home/cbotos/.sbatch-bashrc

cd /home/cbotos/github/qbes/imagenet-exp/
conda activate ffcv

# TEMPLATE END
################################################
# AUTOMATICALLY GENERATED ON:
# {datetime.datetime.now()}
################################################
"""

options = {
    "lr.lr": [1e-4, 1e-3, 1e-2],
    "lr.lr_schedule_type": ["step", "cyclic"],
    "training.batch_size": [512, 256, 128],
    "mlp.trunk_layer": ["first", "last"],
    "mlp.prediction_type": ["regression", "classification"],
    "mlp.balance_weight": [0, 1],
}


launch_commands = ""
for arg, vals in options.items():
    sbatch_content = SBATCH_TEMPLATE
    sbatch_content += f"VALS=({' '.join(map(str, vals))})\n"
    sbatch_content += f"ARG={arg}\n"
    sbatch_content += "VAL=${VALS[$SLURM_ARRAY_TASK_ID]}\n"

    bash_command = f"python train_mlp.py --config swin_configs/swin_K_mlp.yaml "
    bash_command += f"--$ARG $VAL "
    bash_command += '--logging.folder logs/swin/mlp/$ARG"__"$VAL/ '

    sbatch_content += bash_command

    launch_commands += f"sbatch --array=0-{len(vals)-1} --job-name={arg.split('.')[-1]} {SCRIPT_DIR}/{arg}__array.sh\n"
    with open(f"{SCRIPT_DIR}/{arg}__array.sh", "w") as f:
        f.write(sbatch_content)

with open(f"{SCRIPT_DIR}/launch_commands.sh", "w") as f:
    f.write(launch_commands)


