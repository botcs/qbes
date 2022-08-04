python train_MLP.py --config-file rn50_configs/<your config file>.yaml \
    --data.train_dataset=/path/to/train/dataset.ffcv \
    --data.val_dataset=/path/to/val/dataset.ffcv \
    --data.num_workers=12 --data.in_memory=1 \
    --logging.folder=/your/path/here