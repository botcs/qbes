import json
from itertools import combinations

config_dir = "qbes_configs/"

num_blocks = 2 + 2 + 18 + 2
for num_blocks_to_skip in range(num_blocks+1):
    block_ids = range(num_blocks)
    configs = list(combinations(block_ids, num_blocks_to_skip))
    with open(f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}.json", "w") as f:
        json.dump(configs, f, indent=1)

