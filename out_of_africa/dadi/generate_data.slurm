#!/bin/bash
#SBATCH --job-name=generate_out_of_Africa_for_dadi
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
#SBATCH --time=5:00:00

mkdir -p input_fs

source ~/.bashrc && conda activate donni

dem_model="out_of_africa"
file="/xdisk/rgutenk/lnt/software/donni/donni/custom_models.py"
test_data="input_fs"
grids="50 60 70"


# generate test data
echo "Start generating test data for dadi"
echo
date
donni generate_data --model $dem_model --model_file $file --n_samples 30 \
--sample_sizes 20 20 20 --grids $grids --theta 1000 --seed 100 --non_normalize \
--save_individual_fs --outdir $test_data --outfile "${test_data}/test_30_theta_1000"
date
echo "Finish generating test data"
echo
