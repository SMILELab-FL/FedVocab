#!/bin/bash

tasks=$1
model=$2
machine_name=$3
valid_gpu=$4


run_dir=/workspace/code/fedvocab/run/detlm_alone/fedrun_sweep.py

python ${run_dir} \
--tasks ${tasks} \
--model_name ${model} \
--machine_name ${machine_name} \
--valid_gpu ${valid_gpu}
