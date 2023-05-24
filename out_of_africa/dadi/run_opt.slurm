#!/bin/bash
#SBATCH --job-name=dadi_cli_opt_OOA_ns20
#SBATCH --output=outfiles/dadi_opt/%x-%A_%a.out
#SBATCH --error=outfiles/dadi_opt/%x-%A_%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=250gb
#SBATCH --time=72:00:00
#SBATCH --array=0-29

bash dadi_cli_converge $SLURM_ARRAY_TASK_ID