#!/bin/bash
#SBATCH --job-name=split_mig_bench_mark_dadi_160
#SBATCH --output=outfiles/dadi/%x-%A_%a.out
#SBATCH --error=outfiles/dadi/%x-%A_%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=50gb
#SBATCH --time=50:00:00
#SBATCH --array=0-99

source ~/.bashrc && conda activate donni

IN_DIR="input_fs"
OUT_DIR="inference"
mkdir -p ${OUT_DIR}

model="split_mig"
ubounds="1.1e2 1.1e2 3 12 0.5" 
lbounds="1e-3 1e-3 1e-3 1e-3 0"
grids="178 196 214"

i=${SLURM_ARRAY_TASK_ID} # for job array, the fs that we are on
printf -v j "%03d" $i # change $i to $j for index 3

echo "running fs ${j}"
date
dadi-cli InferDM --fs ${IN_DIR}/fs_${j} --model ${model} \
        --grids ${grids} --lbounds ${lbounds} --ubounds ${ubounds} \
        --output ${OUT_DIR}/fs_${j} --cpus 10 \
        --force-convergence --delta-ll .001
date
