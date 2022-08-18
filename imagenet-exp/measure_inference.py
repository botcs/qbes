import gated_swin
import torch 
import json
import time
from tqdm import tqdm

arch = "swin_b"
model_loader = getattr(gated_swin, arch)
weights = getattr(gated_swin, "Swin_B_Weights")
model = model_loader(weights=weights.DEFAULT)

x = torch.randn(1, 3, 224, 224)

# K = [0, 1, 2, 22, 23, 24]
K = [11, 12, 13]



device = "cuda"
x = x.to(device=device)
model.to(device=device)
model.eval()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
with torch.no_grad():
    # Warmup
    for i in tqdm(range(100)):
        _ = model(x)
    stats = torch.load(f"inference_time_stats_{device}.pth")
    for k in K:
        config_file = f"qbes_configs/24nCr{k}-10k.json"
        configs = json.load(open(config_file))
        print(f"K={k}")
        stats[k] = {}
        for config_id, config in tqdm(list(enumerate(configs))):
            run_times = []
            for run_it in range(100):
                start.record()
                _ = model(x, skip_block=config)
                end.record()
                torch.cuda.synchronize()
                run_times.append(start.elapsed_time(end))

            stats[k][config_id] = run_times
        torch.save(stats, f"inference_time_stats_{device}.pth")


device = "cpu"
x = x.to(device=device)
model.to(device=device)
model.eval()

with torch.no_grad():
    # Warmup
    for i in tqdm(range(100)):
        _ = model(x)
    # stats = {}
    stats = torch.load(f"inference_time_stats_{device}.pth")
    for k in K:
        config_file = f"qbes_configs/24nCr{k}.json"
        configs = json.load(open(config_file))
        print(f"K={k}")
        stats[k] = {}
        for config_id, config in tqdm(list(enumerate(configs))):
            run_times = []
            for run_it in range(100):
                start_time = time.time()
                _ = model(x, skip_block=config)
                run_time = time.time() - start_time
                run_times.append(run_time)

            stats[k][config_id] = run_times
        torch.save(stats, f"inference_time_stats_{device}.pth")

    

