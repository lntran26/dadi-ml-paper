#!/bin/bash
#SBATCH --job-name=split_mig_bench_mark_dadi_160_rerun_fs79
#SBATCH --output=outfiles/dadi_2/%x-%A_%a.out
#SBATCH --error=outfiles/dadi_2/%x-%A_%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=50gb
#SBATCH --time=10:00:00

source ~/.bashrc && conda activate donni

IN_DIR="input_fs"
OUT_DIR="inference"

model="split_mig"
ubounds="1.1e2 1.1e2 3 12 0.5" 
lbounds="1e-3 1e-3 1e-3 1e-3 0"
grids="178 196 214"

date
echo "running fs 079"
dadi-cli InferDM --fs ${IN_DIR}/fs_079 --model ${model} \
        --grids ${grids} --lbounds ${lbounds} --ubounds ${ubounds} \
        --output ${OUT_DIR}/fs_079 --optimizations 100000 --cpus 10 \
        --check-convergence --delta-ll .001 \
        --bestfit-p0-file ${OUT_DIR}/fs_079.InferDM.bestfits
date