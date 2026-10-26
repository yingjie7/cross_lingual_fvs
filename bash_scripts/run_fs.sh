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
MODEL=Qwen/Qwen3-8B    # "meta-llama/Meta-Llama-3-8B-Instruct"  # Qwen/Qwen3-8B 

for test_lang in  eng	deu	rus  ;
do
    ### baseline fewshot learning 
    cd ~/multilingual_er/ && python ./src/fs_infer.py --re_gen_data  --model_name Qwen/Qwen3-8B  \
    --lang $test_lang --fs_lang $test_lang --max_new_tokens 10 --shuffle_demons

    for fv_top_n_heads in  {5..20};
    do
        ### fewshot learning + FV 
        cd ~/multilingual_er/ && python ./src/fs_infer.py --re_gen_data  --model_name $MODEL \
        --lang $test_lang --fs_lang $fs_lang  --max_new_tokens 10 \
        --use_fv_on_fs --fv_top_n_heads $fv_top_n_heads 

    done
done 
 
 
wait
