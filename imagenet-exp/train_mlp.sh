python -m ipdb train_mlp.py --config swin_configs/swin_K_mlp.yaml \
    --model.arch "swin_b" \
    --logging.folder=$WORK/qbes/imagenet-test-logs/swin/cache/mlp/ \
    --training.distributed 0 --dist.world_size 1 \
    --data.num_workers=12 --data.in_memory=1 
