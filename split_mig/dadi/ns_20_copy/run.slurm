#!/bin/bash
#SBATCH --job-name=split_mig_bench_mark_dadi_20_force
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
#SBATCH --time=01:00:00
#SBATCH --array=0-9

bash dadi_cli_converge $SLURM_ARRAY_TASK_ID