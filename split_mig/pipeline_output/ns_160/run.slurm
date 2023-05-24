#!/bin/bash
#SBATCH --job-name=donni_split_mig_160_bench_mark_mlpr
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=32gb
#SBATCH --constraint=hi_mem
#SBATCH --time=48:00:00

mkdir -p data
mkdir -p tuned_models
mkdir -p plots

dem_model="split_mig"
grids="178 196 214"

train_data="data/train_5000"
test_data="data/test_1000_theta_1000"
mlpr_dir="tuned_models"
plots_prefix="plots/theta_"

source ~/.bashrc && source activate donni

# generate training data
echo "Start generating training data"
date
donni generate_data --model $dem_model --n_samples 5000 --sample_sizes 160 160 \
--grids $grids --outfile $train_data --seed 1 --generate_tune_hyperparam \
--hyperparam_outfile data/param_dict_tune
date
echo "Finish generating training data"
echo
echo

# tune
echo "Start tuning"
date
donni train --data_file $train_data --mlpr_dir $mlpr_dir --tune --tune_only \
--max_iter 300 --hyperparam data/param_dict_tune
date
echo "Finish tuning"
echo

# train
echo "Start training"
date
donni train --data_file $train_data --mlpr_dir $mlpr_dir \
--hyperparam_list "${mlpr_dir}/tuned_hyperparam_dict_list"
date
echo "Finish training"
echo

# generate test data
echo "Start generating test data"
echo
date
donni generate_data --model $dem_model --n_samples 1000 --sample_sizes 160 160 \
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