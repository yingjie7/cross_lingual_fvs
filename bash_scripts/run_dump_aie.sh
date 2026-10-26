#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 
#!/bin/bash
#SBATCH -J exp-fewshot
#SBATCH -p GPU-1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=phuongnm@jaist.ac.jp 
#SBATCH --gres=gpu:1
# ---------------- 
echo "=== Job info ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Queue:        $SLURM_JOB_NODELIST"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,MaxRSS,NTasks,AllocCPUS,AllocTRES%35,Elapsed

source ~/.bashrc
source ${HOME}/miniconda3/etc/profile.d/conda.sh


# ---------------- 
# ENV
# ---------------- 
PROJECT_PATH=${HOME}/cross_lingual_fvs/

cd $PROJECT_PATH
conda activate ${PROJECT_PATH}/env_crossling_fv/ 

# ---------------- 
# RUN
# ---------------- 

python  src/utils/compute_indirect_effect.py \
 --dataset_name concept_v_object_5 \
 --model_name Qwen/Qwen3-8B 