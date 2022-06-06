import json
from itertools import combinations

from matplotlib.font_manager import json_dump
num_blocks = 16
skip_block_ids = []
for num_drop in [4]:
    block_ids = range(num_blocks)
    skip_block_ids += list(combinations(block_ids, num_drop))

experiments = {k: v for k, v in enumerate(skip_block_ids)}
print(json.dumps(experiments, indent=1))
# print(experiments[0], experiments[1])
    