import json
import random
from itertools import combinations

config_dir = "qbes_configs/"

num_blocks = 2 + 2 + 18 + 2
for num_blocks_to_skip in range(num_blocks+1):
    block_ids = range(num_blocks)
    configs = list(combinations(block_ids, num_blocks_to_skip))

    # Randomly sample 10k
    if len(configs) > 10000:
        random_idxs = random.sample(range(len(configs)), k=10000)
        with open(f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}-10k-indices.json", "w") as f:
            json.dump(random_idxs, f)
            print("saved: ", f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}-10k-indices.json")

        with open(f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}-10k.json", "w") as f:
            configs_subset = [configs[i] for i in random_idxs]
            json.dump(configs_subset, f)
            print("saved: ", f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}-10k.json")

    else:
        with open(f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}.json", "w") as f:
            json.dump(configs, f)
            print("saved: ", f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}-10k.json")
