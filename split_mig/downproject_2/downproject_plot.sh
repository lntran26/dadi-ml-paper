#!/bin/bash

source ~/.bashrc && conda activate donni

mkdir -p plots

python downproject.py 

# donni plot --mlpr_dir "../pipeline_output/ns_20/tuned_models_1" \
# --test_dict "data/projected_test_100_theta_1000_ns39" \
# --model "split_mig" --results_prefix "plots/theta_1000" --coverage --theta 1000


donni plot --mlpr_dir "../pipeline_output/ns_20/tuned_models_1" \
--test_dict "data/projected_test_100_theta_100_ns39" \
--model "split_mig" --results_prefix "plots/theta_100" --coverage --theta 100
