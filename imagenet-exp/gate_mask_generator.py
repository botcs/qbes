import json
from itertools import combinations

num_blocks = 16
num_blocks_to_skip = 7

config_dir = "qbes_configs/"

block_ids = range(num_blocks)
configs = list(combinations(block_ids, num_blocks_to_skip))

# experiments = {k: v for k, v in enumerate(skip_block_ids)}


with open(f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}.json", "w") as f:
    json.dump(configs, f, indent=1)


    