#!/bin/bash
#SBATCH --job-name=generate_split_mig_ns39_downproject
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

source ~/.bashrc && conda activate donni

mkdir -p data

dem_model="split_mig"
test_data="data"
grids="40 50 60"

# generate test data
echo "Start generating test data"
echo
date
donni generate_data --model $dem_model --n_samples 100 --sample_sizes 39 39 \
--grids $grids --theta 1000 --seed 100 \
--outdir $test_data --outfile "${test_data}/test_100_theta_1000_ns39"
date

donni generate_data --model $dem_model --n_samples 100 --sample_sizes 39 39 \
--grids $grids --theta 100 --seed 100 \
--outdir $test_data --outfile "${test_data}/test_100_theta_100_ns39"
date

echo "Finish generating test data"
echo