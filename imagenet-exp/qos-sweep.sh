#!/bin/bash
while true
do 
    for partition in g24 g48
    do
        for qos in low normal high inter
        do
            for jobid in $(squeue --noheader --user $USER --format %A)
            do
                echo scontrol update jobid $jobid qos $qos partition $partition
                scontrol update jobid $jobid qos $qos partition $partition
            done
            sleep 10
        done
    done
done