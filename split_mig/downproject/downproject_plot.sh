#!/bin/bash

source ~/.bashrc && conda activate donni

python downproject.py 

# donni plot --mlpr_dir "../pipeline_output/ns_20/tuned_models_1" \
# --test_dict "data/projected_test_100_theta_1000_ns39" \
# --model "split_mig" --results_prefix "plots/theta_1000" --coverage --theta 1000

# donni plot --mlpr_dir "../pipeline_output/ns_160/tuned_models" \
# --test_dict "data/projected_test_100_theta_1000_ns200" \
# --model "split_mig" --results_prefix "plots/ns200_theta_1000" --coverage --theta 1000

# donni plot --mlpr_dir "../pipeline_output/ns_20/tuned_models_1" \
# --test_dict "data/projected_test_100_theta_100_ns39" \
# --model "split_mig" --results_prefix "plots/theta_100" --coverage --theta 100

# donni plot --mlpr_dir "../pipeline_output/ns_160/tuned_models" \
# --test_dict "data/projected_test_100_theta_100_ns200" \
# --model "split_mig" --results_prefix "plots/ns200_theta_100" --coverage --theta 100

donni plot --mlpr_dir "../pipeline_output/ns_20/tuned_models_1" \
--test_dict "data/projected_test_1000_theta_100_ns39" \
--model "split_mig" --results_prefix "plots/ns39_1000_theta_1000" --coverage --theta 100

donni plot --mlpr_dir "../pipeline_output/ns_20/tuned_models_1" \
--test_dict "../pipeline_output/ns_20/data/test_100_theta_100" \
--model "split_mig" --results_prefix "plots/ns20_100_theta_100" --coverage --theta 100