#!/bin/bash
#SBATCH --job-name=tune_pgnn   # name that you chose
#SBATCH -A watertemp           # your account
#SBATCH -p UV                  # the partition you want to use - for GPUs, use UV
#SBATCH -c 1                   # number of cores per task
#SBATCH --gres=gpu:tesla:1     # specify how many GPU cores you want, and what kind (tesla or quadro)
#SBATCH --time=00:10:00        # time at which the process will be cancelled if unfinished
#SBATCH -o log/tune-tesla/%A_%a.log  # sets output log file to array_job.log
#SBATCH -e log/tune-tesla/%A_%a.err  # sets error log file to array_job.err
#SBATCH --export=ALL
#SBATCH --array=0-2            # process IDs corresponding to list order (0-indexed) of model_params.npy

cd /cxfs/projects/usgs/water/iidd/data-sci/lake-temp/lake-temperature-neural-networks-apa
module load python/anaconda3 cuda/9.0.176
source activate tensorflow
python run_job.py ${SLURM_ARRAY_TASK_ID}
source deactivate

# Usage:
# sbatch -a 0-1 slurm_jobs.sh
# sbatch slurm_jobs.sh
