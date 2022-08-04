import torch
import json
import tqdm
import math
import os


import argparse
p = argparse.ArgumentParser()
p.add_argument("K", nargs="+", type=int)
args = p.parse_args()

K = args.K

def file_count(root_dir):
    import glob
    fnames = glob.glob(f"{root_dir}/[0-9]*.pth")
    return len(fnames)


def concatenate_files(root_dir):
    import glob
    fnames = glob.glob(f"{root_dir}/[0-9]*.pth")
    print("total files found: ", len(fnames))
    
    fnames = sorted(fnames)
    preds = []
    targets = []
    inference_times = []
    for fname in tqdm.tqdm(fnames):
        f = torch.load(fname)
        p = f["preds"]
        t = f["targets"]
        inf_t = f["inference_time"]
        preds.append(p)
        targets.append(t)
        inference_times.append(inf_t)

    if len(preds) == 0:
        return
    preds = torch.stack(preds, dim=0)
    targets = torch.stack(targets, dim=0)
    inference_times = torch.tensor(inference_times)

    f = {
        "preds": preds,
        "targets": targets,
        "inference_times": inference_times
    }
    torch.save(f, f"{root_dir}/all.pth")


train_missdict = {}
for k in K:
    config_indices=json.load(open(f"qbes_configs/24nCr{k}-10k-indices.json"))
    num_configs = len(config_indices)
    fc = file_count(f"./logs/swin/cache/val/full/24nCr{k}{'-10k' if (k>4 and k<20) else ''}/")
    print(f"{k:2d}: {fc:5} out of {num_configs}")
    if fc != num_configs:
        train_missdict[k] = [j for j in config_indices if not os.path.exists(f"./logs/swin/cache/val/full/24nCr{k}/{j:05d}.pth")]
    concatenate_files(f"./logs/swin/cache/val/full/24nCr{k}{'-10k' if (k>4 and k<20) else ''}/")


val_missdict = {}
for k in K:
    config_indices = json.load(open(f"qbes_configs/24nCr{k}-10k-indices.json"))
    num_configs = len(config_indices)
    fc = file_count(f"./logs/swin/cache/train/0.1/24nCr{k}{'-10k' if (k>4 and k<20) else ''}/")
    print(f"{k:2d}: {fc:5} out of {num_configs}")
    if fc != num_configs:
        val_missdict[k] = [j for j in config_indices if not os.path.exists(f"./logs/swin/cache/train/0.1/24nCr{k}/{j:05d}.pth")]
    concatenate_files(f"./logs/swin/cache/train/0.1/24nCr{k}{'-10k' if (k>4 and k<20) else ''}/")