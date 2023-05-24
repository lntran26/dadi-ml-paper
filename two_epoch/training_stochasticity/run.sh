#!/bin/bash

date
source ~/.bashrc && conda activate donni
run_id=$RANDOM

dem_model="two_epoch"
train_data="../pipeline_output/data/train_5000"
test_data="../pipeline_output/data/test_1000_theta_1000"
tuned_results="../pipeline_output/tuned_models/tuned_hyperparam_dict_list"
mlpr_dir="tuned_models_${run_id}"
# plots_prefix="plots/run_${run_id}_theta_"
plots_prefix="plots/more_stochasticity/run_${run_id}_theta_"

mkdir -p $mlpr_dir

# train
echo "Start training"
donni train --data_file $train_data --mlpr_dir $mlpr_dir --max_iter 300 \
--hyperparam_list $tuned_results
echo "Finish training"
echo

# plot
echo "Plotting"
donni plot --mlpr_dir $mlpr_dir --test_dict $test_data \
--results_prefix "${plots_prefix}1000" --model $dem_model --coverage --theta 1000
echo
echo "Finish run"

date