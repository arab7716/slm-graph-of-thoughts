#!/bin/bash
#SBATCH --job-name=got-eval-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00   # Max time for the longest single task
#SBATCH --array=0-5       # Spin up 6 parallel jobs!
#SBATCH --exclude=neu306
#SBATCH --output=slurm_logs/eval-%A_%a.out

# Slurm script to run all tasks (ablations can be toggled within the files themselves)
# Code syntax generation partially aided by generative AI. 
# All code thoroughly human reviewed.

# --- PARALLEL MATRIX SETUP ---
# Job 0: Sorting (0.6)
# Job 1: Sorting (0.1)
# Job 2: Keyword (0.6)
# Job 3: Keyword (0.1)
# Job 4: Sets (0.6)
# Job 5: Sets (0.1)

TASKS=("sorting" "sorting" "keyword" "keyword" "sets" "sets")
TEMPS=(0.6 0.1 0.6 0.1 0.6 0.1)
MODELS=("qwen2.5-14b" "qwen2.5-14b" "qwen2.5-14b" "qwen2.5-14b" "qwen2.5-14b" "qwen2.5-14b")

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
TEMP=${TEMPS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================"
echo "STARTING JOB ID: $SLURM_ARRAY_TASK_ID"
echo "TASK: $TASK | MODEL: $MODEL_NAME | TEMP: $TEMP"
echo "======================================================"

# --- CACHE SETUP ---
PERMANENT_CACHE="/n/fs/fail-llm/huggingface_cache"
TEMP_CACHE="/scratch/$USER/got_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TEMP_CACHE"

echo "[$(date +'%H:%M:%S')] Staging models to scratch disk..."
rsync -avh --prune-empty-dirs "$PERMANENT_CACHE/" "$TEMP_CACHE/"

TEMP_CONFIG="config_job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
sed "s#$PERMANENT_CACHE#$TEMP_CACHE#g" config.json > "$TEMP_CONFIG"

# --- ENVIRONMENT ---
source /n/fs/fail-llm/miniconda3/etc/profile.d/conda.sh
conda activate llama_test_env


# --- DYNAMIC TASK ROUTING ---

if [ "$TASK" == "sorting" ]; then
    echo "--- RUNNING SORTING 24 ---"
    START_TIME=$(date +%s)
    python -m examples.sorting.sorting_024 \
        --config_path "$TEMP_CONFIG" \
        --model_name "$MODEL_NAME" \
        --temperature "$TEMP"
    END_TIME=$(date +%s)

elif [ "$TASK" == "keyword" ]; then
    echo "--- RUNNING KEYWORD COUNTING ---"
    START_TIME=$(date +%s)
    python -m examples.keyword_counting.keyword_counting \
        --config_path "$TEMP_CONFIG" \
        --model_name "$MODEL_NAME" \
        --temperature "$TEMP"
    END_TIME=$(date +%s)

elif [ "$TASK" == "sets" ]; then
    echo "--- RUNNING SET INTERSECTION ---"
    START_TIME=$(date +%s)
    python -m examples.set_intersection.set_intersection_032 \
        --config_path "$TEMP_CONFIG" \
        --model_name "$MODEL_NAME" \
        --temperature "$TEMP"
    END_TIME=$(date +%s)
fi

# --- FINAL REPORT & CLEANUP ---
TOTAL_TIME=$((END_TIME - START_TIME))
echo ""
echo "======================================================"
echo "TASK COMPLETED: $TASK at Temp $TEMP"
echo "TOTAL EXECUTION TIME: $TOTAL_TIME seconds"
echo "======================================================"

echo "[$(date +'%H:%M:%S')] Cleaning up scratch disk..."
rm -rf "$TEMP_CACHE"
rm "$TEMP_CONFIG"
echo "Job Done."


