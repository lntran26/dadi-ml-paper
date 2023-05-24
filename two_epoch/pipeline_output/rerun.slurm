#!/bin/bash

#SBATCH --job-name=run_donni_pipeline_two_epoch_unfolded_20_new_bound_retune
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=50gb
#SBATCH --time=00:05:00

date

dem_model="two_epoch"
train_data="data/train_5000"
test_data="data/test_1000_theta_1000"
mlpr_dir="tuned_models"
plots_prefix="plots/theta_"

# tune
echo "Start tuning"
date
donni train --data_file $train_data --mlpr_dir $mlpr_dir --tune \
--max_iter 300 --hyperparam data/param_dict_tune
date
echo "Finish tuning & training"
echo

# train
# echo "Start training"
# date
# donni train --data_file $train_data --mlpr_dir $mlpr_dir --max_iter 300 \
# --hyperparam_list tuned_models/tuned_hyperparam_dict_list
# date
# echo "Finish training"
# echo

# plot
echo "Plotting"
donni plot --mlpr_dir $mlpr_dir --test_dict $test_data \
--results_prefix "${plots_prefix}1000" --model $dem_model --coverage --theta 1000
echo
echo "Finish run"

date