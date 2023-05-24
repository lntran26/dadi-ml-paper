#!/bin/bash
#SBATCH --job-name=dadi_cli_opt_OOA_ns20_rep
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
#SBATCH --array=000,004,011,023,024,025,026,028

bash dadi_cli_converge_best_fit $SLURM_ARRAY_TASK_ID