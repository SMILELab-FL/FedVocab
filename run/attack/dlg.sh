#!/bin/bash

dirs=$1
times=$2
model_type=$3
index=$4
bs=$5

model_name=${model_type}-base-uncased
run_dir=${dirs}/code/fedvocab/run/attack/dlg.py

python ${run_dir} \
--mode fedde \
--model_type ${model_type} \
--index ${index} \
--bs ${bs} \
--output_path ${dirs}/output/fednlp/attack/inversion_output/ \
--raw_path ${dirs}/output/fednlp/attack/agnews_raw.text \
--opp ${dirs}/pretrain/nlp/${model_name}/ \
--omp ${dirs}/pretrain/nlp/${model_name}/ \
--trained_model_path ${dirs}/output/fednlp/agnews/feddea/tlm/after/ \
--times ${times}
