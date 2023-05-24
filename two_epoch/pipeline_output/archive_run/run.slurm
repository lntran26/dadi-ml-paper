#!/bin/bash

#SBATCH --job-name=run_donni_pipeline_two_epoch_unfolded_20_new_bound
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

source ~/.bashrc && conda activate donni

mkdir -p data
mkdir -p tuned_models
mkdir -p plots

dem_model="two_epoch"
train_data="data/train_5000"
test_data="data/test_1000_theta_1000"
grids="24 28 32"
mlpr_dir="tuned_models"
plots_prefix="plots/theta_"

# generate training data
echo "Start generating training data"
date
donni generate_data --model $dem_model --n_samples 5000 --sample_sizes 20 \
--grids $grids --outfile $train_data --seed 1 --generate_tune_hyperparam \
--hyperparam_outfile data/param_dict_tune
date
echo "Finish generating training data"
echo

# tune
echo "Start tuning"
date
donni train --data_file $train_data --mlpr_dir $mlpr_dir --tune \
--max_iter 300 --hyperparam data/param_dict_tune
date
echo "Finish tuning"
echo

# generate test data
echo "Start generating test data"
echo
date
donni generate_data --model $dem_model --n_samples 1000 --sample_sizes 20 \
--grids $grids --theta 1000 --seed 100 --outfile $test_data
date
echo "Finish generating test data"
echo

# plot
echo "Plotting"
donni plot --mlpr_dir $mlpr_dir --test_dict $test_data \
--results_prefix "${plots_prefix}1000" --model $dem_model --coverage --theta 1000
echo
echo "Finish run"

date