#!/bin/bash
#SBATCH --job-name=dadi_cli_opt_OOA_ns20_rep_fs23
#SBATCH --output=outfiles/dadi_opt_2/%x-%A_%a.out
#SBATCH --error=outfiles/dadi_opt_2/%x-%A_%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=250gb
#SBATCH --time=24:00:00

source ~/.bashrc && conda activate donni

IN_DIR="input_fs"
OUT_DIR="inference"

model_file="/xdisk/rgutenk/lnt/software/donni/donni/custom_models.py"
model="out_of_africa"

# six nu, four m, three T, one misid
ubounds="1.1e2 1.1e2 1.1e2 1.1e2 1.1e2 1.1e2 11 11 11 11 2.1 2.1 2.1 1" 
lbounds="1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 0 0 0 0 1e-3 1e-3 1e-3 1e-5"
grids="40 50 60"

date
echo "running fs 023"
dadi-cli InferDM --fs ${IN_DIR}/fs_023 --model ${model} --model-file ${model_file} \
        --grids ${grids} --lbounds ${lbounds} --ubounds ${ubounds} \
        --output-prefix ${OUT_DIR}/fs_023 --optimizations 100000 --cpus 50 \
        --check-convergence --delta-ll .0005 \
        --bestfit-p0-file ${OUT_DIR}/fs_023.InferDM.bestfits
date