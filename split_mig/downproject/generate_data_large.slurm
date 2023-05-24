#!/bin/bash
#SBATCH --job-name=generate_split_mig_ns200_downproject
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=100gb
#SBATCH --time=5:00:00

source ~/.bashrc && conda activate donni

dem_model="split_mig"
test_data="data"
grids="222 244 266"


# generate test data
echo "Start generating test data for down projection test"
echo
date
donni generate_data --model $dem_model --n_samples 100 --sample_sizes 200 200 \
--grids $grids --theta 1000 --seed 100 \
--outdir $test_data --outfile "${test_data}/test_100_theta_1000_ns200"
date
echo "Finish generating test data"
echo