#!/usr/bin/env python
# coding: utf-8

import torch
import json
from tqdm import tqdm
import os





num_classes = 1000
def build_stats(log_fname, proportions):
    torch.cuda.empty_cache()
    store_device="cpu"
    compute_device="cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"store device: {store_device}")
    print(f"compute device: {compute_device}")
    dtype=torch.float32
    config_results = torch.load(log_fname)
    num_configs = len(config_results["preds"])
    preds = config_results["preds"].to(dtype=dtype, device=compute_device)
    targets = config_results["targets"].to(dtype=dtype, device=compute_device)
    
    prec = torch.zeros(len(proportions), num_classes, num_configs, dtype=dtype, device=store_device)
    rec = torch.zeros(len(proportions), num_classes, num_configs, dtype=dtype, device=store_device)
    f1 = torch.zeros(len(proportions), num_classes, num_configs, dtype=dtype, device=store_device)
    
    TP = preds == targets
    for class_id in tqdm(range(num_classes)):
        class_pred_mask = preds == class_id
        class_target_mask = targets == class_id
        class_TP = TP * class_target_mask
        # class_TN = ~class_pred_mask * ~class_target_mask
        
        prev_size = 0
        TP_sum = 0
        class_TP_sum = 0
        class_TN_sum = 0
        class_pred_sum = 0
        class_target_sum = 0
        # Use cumulative window sliding to avoid recomputing the sums
        for proportion_id, proportion in enumerate(proportions):
            torch.cuda.empty_cache()
            size = int(preds.shape[1] * proportion)
            # Total numbers of True Positives of this class so far
            class_TP_sum += class_TP[:, prev_size:size].sum(dim=1)
            
            # Total numbers of positive predictions of this class so far
            class_pred_sum += class_pred_mask[:, prev_size:size].sum(dim=1)
            
            # Total numbers of actuall positive samples of this class so far
            class_target_sum += class_target_mask[:, prev_size:size].sum(dim=1)
            
            # METRICS COMPUTATION
            class_prec = class_TP_sum / torch.clamp(class_pred_sum, 1)
            class_rec = class_TP_sum / torch.clamp(class_target_sum, 1)
            class_f1 = 2 * (class_prec * class_rec) / torch.clamp(class_prec + class_rec, 1)
            
            prec[proportion_id, class_id] = class_prec.to(device=store_device)
            rec[proportion_id, class_id] = class_rec.to(device=store_device)
            f1[proportion_id, class_id] = class_f1.to(device=store_device)
            
            prev_size = size
    
    stats = {
        "prec": prec.cpu(), 
        "rec": rec.cpu(),
        "f1": f1.cpu(),
    }
    return stats


K = list(range(25))
proportions = torch.linspace(.05, 1, 20)
for k in K:
    print(f"Aggregating validation statistics for K={k}")
    if os.path.exists(f"notebook-cache/swin/train_stats_k{k:02d}.pth"):
        print(f"EXISTING FILE: notebook-cache/swin/train_stats_k{k:02d}.pth - skipping...")
        continue
    train_stats = build_stats(
        f"./logs/swin/cache/train/0.1/24nCr{k}{'-10k' if (k>4 and k<20) else ''}/all.pth", 
        proportions=proportions
    )
    torch.save(train_stats, f"notebook-cache/swin/train_stats_k{k:02d}.pth")


# In[30]:



for k in K:
    print(f"Aggregating validation statistics for K={k}")
    if os.path.exists(f"notebook-cache/swin/val_stats_k{k:02d}.pth"):
        print(f"EXISTING FILE: notebook-cache/swin/val_stats_k{k:02d}.pth - skipping...")
        continue
    val_stats = build_stats(
        f"./logs/swin/cache/val/full//24nCr{k}{'-10k' if (k>4 and k<20) else ''}/all.pth", 
        proportions=[1.0]
    )
    torch.save(val_stats, f"notebook-cache/swin/val_stats_k{k:02d}.pth")


# In[ ]:




