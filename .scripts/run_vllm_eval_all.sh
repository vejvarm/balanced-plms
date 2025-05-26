#!/bin/bash
# [model_name] [num_runs] [from_cache]
cd ~/git/balanced-plms/
bash run_ssc_vllm_eval.sh "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 5 true
bash run_ssc_vllm_eval.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 5 true
bash run_ssc_vllm_eval.sh "meta-llama/Llama-3.1-8B-Instruct" 5 true
bash run_ssc_vllm_eval.sh "Qwen/Qwen2.5-7B-Instruct-1M" 5 true
bash run_ssc_vllm_eval.sh "Qwen/Qwen2.5-14B-Instruct-1M" 5 true # DONE
# bash run_ssc_vllm_eval.sh "meta-llama/CodeLlama-7b-hf" 5  # FAIL!
bash run_ssc_vllm_eval.sh "meta-llama/CodeLlama-7b-Instruct-hf" 5 true
bash run_ssc_vllm_eval.sh "meta-llama/CodeLlama-13b-Instruct-hf" 5 true
# bash run_ssc_vllm_eval.sh "meta-llama/Llama-2-7b-chat-hf" 5 true # DONE
bash run_ssc_vllm_eval.sh "meta-llama/Llama-2-13b-chat-hf" 5 true