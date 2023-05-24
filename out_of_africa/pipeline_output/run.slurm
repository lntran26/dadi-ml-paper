#!/bin/bash
#SBATCH --job-name=donni_out_of_Africa
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=250gb
#SBATCH --time=48:00:00

mkdir -p data
mkdir -p tuned_models
mkdir -p plots

source ~/.bashrc && conda activate donni

dem_model="out_of_africa"
file="/xdisk/rgutenk/lnt/software/donni/donni/custom_models.py"
train_data="data/train_5000"
test_data="data/test_1000_theta_1000"
grids="50 60 70"
mlpr_dir="tuned_models"
plots_prefix="plots/theta_"

# generate training data
echo "Start generating training data"
date
donni generate_data --model $dem_model --model_file $file --n_samples 5000 \
--sample_sizes 20 20 20 --grids $grids --outfile $train_data --seed 1 \
--generate_tune_hyperparam --hyperparam_outfile data/param_dict_tune
date
echo "Finish generating training data"
echo


# generate test data
echo "Start generating test data"
echo
date
donni generate_data --model $dem_model --model_file $file --n_samples 1000 \
--sample_sizes 20 20 20 --grids $grids --theta 1000 --seed 100 \
--outdir $test_data --outfile $test_data
date
echo "Finish generating test data"
echo


# tune
echo "Start tuning and training"
date
donni train --data_file $train_data --mlpr_dir $mlpr_dir --tune \
--max_iter 300 --hyperparam data/param_dict_tune
date
echo "Finish tuning and training"
echo


# plot
echo "Plotting"
donni plot --mlpr_dir $mlpr_dir --test_dict $test_data --model_file $file \
--model $dem_model --results_prefix "${plots_prefix}1000" --coverage --theta 1000
echo
echo "Finish run"

date
