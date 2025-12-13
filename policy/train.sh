#!/bin/bash
#SBATCH --job-name=e2e_vla_metaworld-tiny                   
#SBATCH --output="e2e_vla_metaworld-tiny-%j.out" 
#SBATCH --nodes=1                         
#SBATCH --ntasks=1                        
#SBATCH --cpus-per-task=8                 
#SBATCH --gres=gpu:nvidia-rtx-a6000:1                      
#SBATCH --partition=gpu                   
#SBATCH --mem-per-cpu=2G

source venv/bin/activate

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


echo "Starting End-to-End VLA Training..."
echo "Task: pick-place"
echo "Start Time: $(date)"
echo "============================================"

python tiny_train_e2e.py

echo "============================================"
echo "Training completed!"
echo "End Time: $(date)"
echo "============================================"

deactivate
