import json
from itertools import combinations

config_dir = "qbes_configs/"

num_blocks = 16
for num_blocks_to_skip in [0, 3, 4, 5, 6, 7]:
    block_ids = range(num_blocks)
    configs = list(combinations(block_ids, num_blocks_to_skip))
    with open(f"{config_dir}/{num_blocks}nCr{num_blocks_to_skip}.json", "w") as f:
        json.dump(configs, f, indent=1)


    