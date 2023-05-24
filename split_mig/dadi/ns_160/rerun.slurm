#!/bin/bash
#SBATCH --job-name=split_mig_bench_mark_dadi_160_rerun
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
#SBATCH --array=002,007,008,012,013,020,022,024,031,034,037,045,047,051,054,055,057,058,060,061,064,068,069,072,079,081,088,089,093,094,098,099


bash dadi_cli_converge_best_fit $SLURM_ARRAY_TASK_ID
