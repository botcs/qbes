#!/bin/bash
K=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
num_tasks=(1 16 120 560 1820 4368 8008 11440 12870 11440 8008 4368 1820 560 120 16 1)
chunk_size=100
for i in {0..16}
do
    # at this point it should be clear to the reader
    # that I clearly have no clue how to use bash
    k=${K[$i]}
    n=${num_tasks[$i]}
    num_chunks=$((n / chunk_size))
    echo sbatch --array=0-$num_chunks eval-cluster/training-set-0.1/chooseK.sh $k
    # sbatch --array=0-$num_chunks eval-cluster/training-set-0.1/chooseK.sh $k
done