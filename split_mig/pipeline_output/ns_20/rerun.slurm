#!/bin/bash
#SBATCH --job-name=donni_split_mig_20_rerun
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
#SBATCH --time=00:05:00

# rerun because the original run.slurm script has a bug in the train data path.
(It has been updated to fix the bug.)

dem_model="split_mig"

train_data="data/train_5000_theta_"
test_data="data/test_"
mlpr_dir="tuned_models_"
plots_prefix="plots/theta_"

source ~/.bashrc && source activate donni

# train
echo "Start training for benchmarking"
date
donni train --data_file "${train_data}1" --mlpr_dir "${mlpr_dir}1" \
--hyperparam_list "${mlpr_dir}1/tuned_hyperparam_dict_list"
date
echo "Finish training for benchmarking"
echo

# plot
echo "Plotting for models trained on theta 1 and test on theta 1000"
donni plot --mlpr_dir "${mlpr_dir}1" --test_dict "${test_data}1000_theta_1000" \
--model $dem_model --results_prefix "${plots_prefix}1000" --coverage --theta 1000
echo
echo "Finish run"

date