#!/bin/bash
#SBATCH --job-name=generate_split_mig_20_for_dadi
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

mkdir -p input_fs

source ~/.bashrc && conda activate donni

dem_model="split_mig"
test_data="input_fs"
grids="40 50 60"


# generate test data
echo "Start generating test data for dadi"
echo
date
donni generate_data --model $dem_model --n_samples 100 --sample_sizes 20 20 \
--grids $grids --theta 1000 --seed 100 --non_normalize --save_individual_fs \
--outdir $test_data --outfile "${test_data}/test_100_theta_1000"
date
echo "Finish generating test data"
echo