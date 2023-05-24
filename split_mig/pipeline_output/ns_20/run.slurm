#!/bin/bash
#SBATCH --job-name=donni_split_mig_20
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
#SBATCH --time=24:00:00

mkdir -p data
mkdir -p plots

dem_model="split_mig"
grids="40 50 60"
g10="13 16 19"
g20="24 28 32"
g40="46 52 58"
g80="90 100 110"
g160="178 196 214"

train_data="data/train_5000_theta_"
test_data="data/test_"
mlpr_dir="tuned_models_"
plots_prefix="plots/theta_"

source ~/.bashrc && source activate donni

# generate test data 1000
echo "Start generating test data 1000 for benchmark"
echo
date
donni generate_data --model $dem_model --n_samples 1000 --sample_sizes 20 20 \
--grids $grids --theta 1000 --seed 100 --outfile "${test_data}1000_theta_1000"
date
echo "Finish generating test data 1000 for benchmark"
echo

# generate test data 100
echo "Start generating test data 100 for noise experiment"
echo
date
for i in 1 10000 1000 100
do
    donni generate_data --model $dem_model --n_samples 100 --sample_sizes 20 20 \
    --grids $grids --theta $i --seed 100 --outfile "${test_data}100_theta_${i}"
done
date
echo "Finish generating test data 100 for noise experiment"
echo

# generate 5000 training data theta 1 for benchmark
echo "Start generating training data theta 1 for benchmark"
date
donni generate_data --model $dem_model --n_samples 5000 --sample_sizes 20 20 \
--grids $grids --outfile "${train_data}1" --seed 1 --generate_tune_hyperparam \
--hyperparam_outfile data/param_dict_tune
date
echo "Finish generating 5000 training data theta 1"
echo
echo

# generate training data for the rest of noise experiment
echo "Start generating training data theta 10000 1000 and 100 for noise experiment"
date
for i in 10000 1000 100
do
    echo "Start generating ${train_data}${i}"
    date
    donni generate_data --model $dem_model --n_samples 5000 --sample_sizes 20 20 \
    --grids $grids --theta $i --outfile "${train_data}${i}" --seed 1
    date
done
date
echo "Finish generating 5000 training data theta 10000 1000 and 100"
echo

# make directories for different tuned models
for i in 1 10000 1000 100
do
    mkdir -p "${mlpr_dir}${i}"
done

# tune
echo "Start tuning with training data theta 1 for benchmarking"
date
donni train --data_file "${train_data}1" --mlpr_dir "${mlpr_dir}1" --tune --tune_only \
--max_iter 300 --hyperparam data/param_dict_tune
date
echo "Finish tuning for benchmarking"
echo

# train
echo "Start training for benchmarking"
date
donni train --data_file "${train_data}1" --mlpr_dir "${mlpr_dir}1" \
--hyperparam_list "${mlpr_dir}1/tuned_hyperparam_dict_list"
date
echo "Finish training for benchmarking"
echo

# tune and train for noise experiment
for i in 10000 1000 100
do
    echo "Start tuning and training using ${train_data}${i}"
    date
    donni train --data_file "${train_data}${i}" --mlpr_dir "${mlpr_dir}${i}" \
    --tune --max_iter 300 --hyperparam data/param_dict_tune
    date
done
echo "Finish tuning and training"
echo

# plot
echo "Plotting for models trained on theta 1 and test on theta 1000"
donni plot --mlpr_dir "${mlpr_dir}1" --test_dict "${test_data}1000_theta_1000" \
--model $dem_model --results_prefix "${plots_prefix}1000" --coverage --theta 1000
echo
echo "Finish run"

date