#!/bin/bash
K=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
num_tasks=(1 24 276 2024 10626 42504 134596 346104 735471 1307504 1961256 2496144 2704156 2496144 1961256 1307504 735471 346104 134596 42504 10626 2024 276 24 1)
chunk_size=20
for i in {0..24}
do
    # at this point it should be clear to the reader
    # that I clearly have no clue how to use bash
    k=${K[$i]}
    n=${num_tasks[$i]}
    num_chunks=$((n / chunk_size))
    # Reduce the search space to the last 10k configs
    array_from_id=$(((n-10000) / chunk_size))
    # max(x, 0) - clamp at 0
    array_from_id=$((array_from_id > 0 ? array_from_id : 0))
    array_to_id=$num_chunks
    echo sbatch --array=$array_from_id-$array_to_id --job-name=\"K$k-train\" eval-cluster/training-set-0.1/chooseK-swin.sh $k $chunk_size
done