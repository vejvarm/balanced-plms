#!/bin/bash
# run_ssc_vllm_eval.sh

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_name_or_path> <n_runs> [<from_cache>]" 
    exit 1
fi

MODEL_NAME="$1"
N_RUNS="$2"
FROM_CACHE="$3"

# Calculate MAX_NUM_SEQS as half of N_RUNS (integer division)
MAX_NUM_SEQS=$((N_RUNS / 2))

# Determine optional flag for --from-cache
FROM_CACHE_ARG=""
if [ -n "$FROM_CACHE" ]; then
    if [ "$FROM_CACHE" = "true" ]; then
        FROM_CACHE_ARG="--from-cache"
    fi
fi

# ______ON SUBSEQUENT RUNS__
docker start rdf4j_server
docker start neo4j_server

cd /work/models
source ~/miniconda3/bin/activate vllm

screen -XS vllm_server kill
sleep 10
screen -dmS vllm_server bash -c "vllm serve $MODEL_NAME --api-key token-abc123 --enable-chunked-prefill --max-model-len 5000 --gpu-memory-utilization 0.9 --download-dir ./"

echo "Waiting for vllm to initialize..."
sleep 5
echo "vllm should be running now."

cd ~/git/balanced-plms/
source ~/miniconda3/bin/activate ./.conda

CONFIG_BASE="configs/3_eval_vllm_templates"

# Cleanup leftover temporary config files.
find "$CONFIG_BASE" -type f -name '*_temp_run*.json' -delete

find "$CONFIG_BASE" -type f -name "*.json" | while read config; do
    rel_path="${config#$CONFIG_BASE/}"
    rel_path_no_ext="${rel_path%.json}"
    out_dir="./results/${MODEL_NAME}/ssc/${rel_path_no_ext}"
    mkdir -p "$out_dir"
    
    for run in $(seq 1 "$N_RUNS"); do
        new_config="${config%.json}_temp_run${run}.json"
        jq --arg model "$MODEL_NAME" --arg output "${out_dir}/evaluation_results_run${run}" \
           '.model_name_or_path = $model | .output_dir = $output' \
           "$config" > "$new_config"
    
        echo "Running evaluation with config: $new_config (run: $run, output: ${out_dir}/evaluation_results_run${run})"
        python seq2seq/eval_vllm.py "$new_config" $FROM_CACHE_ARG &
        sleep 1
    done
    wait
    rm "${config%.json}_temp_run"*.json
done
