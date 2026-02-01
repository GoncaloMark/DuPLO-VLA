#!/bin/bash
#SBATCH --job-name=metaworld_data_gen             
#SBATCH --output="metaworld_data_gen-%j.out" 
#SBATCH --nodes=1                         
#SBATCH --ntasks=1                        
#SBATCH --cpus-per-task=8                 
#SBATCH --gres=gpu:nvidia-rtx-a6000:1                      
#SBATCH --partition=gpu                   
#SBATCH --mem-per-cpu=2G

source /data/home/g.marques/storage/DuPLO-VLA/venv/bin/activate

echo "============================================"
echo "SLURM Job Information"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Temporary Directory: $TMPDIR"
echo "============================================"

echo "Python Environment:"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo "============================================"


echo "Starting Metaworld Data Generation..."
echo "Start Time: $(date)"
echo "============================================"

./gen_multitask_ds.sh

echo "============================================"
echo "Generation completed!"
echo "End Time: $(date)"
echo "============================================"

echo "Starting Metaworld Data Augmentation..."
echo "Start Time: $(date)"
echo "============================================"

./gen_augment_ds.sh

echo "============================================"
echo "Augmentation completed!"
echo "End Time: $(date)"
echo "============================================"

echo "Collapsing Metaworld Data To Single Dataset..."
echo "Start Time: $(date)"
echo "============================================"

python collapse_ds.py

echo "Collapsing Completed!"
echo "End Time: $(date)"
echo "============================================"

deactivate
