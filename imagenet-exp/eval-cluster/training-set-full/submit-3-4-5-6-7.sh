#!/bin/bash
num_tasks=(560 1820 4368 8008 11440)
K=(3 4 5 6 7)
chunk_size=50
for i in {0..4}
do
    # at this point it should be clear to the reader
    # that I clearly have no clue how to use bash
    n=${num_tasks[$i]}
    k=${K[$i]}
    num_chunks=$((n / chunk_size))
    echo --array=0-$num_chunks eval-cluster/training-set-full/chooseK.sh $k
    sbatch --array=0-$num_chunks eval-cluster/training-set-full/chooseK.sh $k
done