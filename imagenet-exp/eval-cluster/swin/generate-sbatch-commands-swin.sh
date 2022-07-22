#!/bin/bash
K=(0 1 2 3 4 20 21 22 23 24)
num_tasks=(1 24 276 2024 10626 10626 2024 276 24 1)
chunk_size=20
for i in {0..9}
do
    # at this point it should be clear to the reader
    # that I clearly have no clue how to use bash
    k=${K[$i]}
    n=${num_tasks[$i]}
    num_chunks=$((n / chunk_size))
    # Reduce the search space to the last 10k configs

    array_from_id=0
    array_to_id=$num_chunks 
    echo sbatch --array=$array_from_id-$array_to_id --job-name=\"K$k-train\" eval-cluster/training-set-0.1/chooseK-swin.sh $k $chunk_size
    echo sbatch --array=$array_from_id-$array_to_id --job-name=\"K$k-val\" eval-cluster/validation-set-full/chooseK-swin.sh $k $chunk_size
done

echo "CLAMPED @ 10k"
# Only run 10k configs for these
K=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
num_tasks=10000
chunk_size=20
for i in {0..14}
do
    # at this point it should be clear to the reader
    # that I clearly have no clue how to use bash
    k=${K[$i]}
    n=$num_tasks
    num_chunks=$((n / chunk_size))

    array_from_id=0
    array_to_id=$num_chunks
    echo sbatch --array=$array_from_id-$array_to_id --job-name=\"K$k-train\" eval-cluster/training-set-0.1/chooseK-swin.sh $k $chunk_size
    echo sbatch --array=$array_from_id-$array_to_id --job-name=\"K$k-val\" eval-cluster/validation-set-full/chooseK-swin.sh $k $chunk_size
done