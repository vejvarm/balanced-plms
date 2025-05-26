cd /work/models
source ~/miniconda3/bin/activate vllm

screen -XS vllm_server kill
screen -dmS vllm_server vllm serve meta-llama/Llama-3.1-8B-Instruct --api-key token-abc123 --enable-chunked-prefill --max-model-len 2000 --max-num-seqs 2 --gpu-memory-utilization 0.8 --download-dir ./

echo "Waiting for vllm to initialize..."
sleep 120  # wait for 2 minutes
echo "vllm should be running now."

cd ~/git/balanced-plms/
source ~/miniconda3/bin/activate ./.conda

# ______ON FIRST RUN______:
# # serve RDF4j server in docker
# docker run --name rdf4j_server -d -p 8181:8080 eclipse/rdf4j-workbench:latest

# # serve Neo4j server in docker
# export groups=( $( id --real --groups neo4j ) )
# docker run \
#     --name neo4j_server \
#     --user="$(id -u neo4j):$(id -g neo4j)" \
#     --group-add=$groups \
#     -p7474:7474 -p7687:7687 \
#     -d \
#     -v /neo4j/data:/data \
#     -v /neo4j/logs:/logs \
#     -v /neo4j/import:/import \
#     -v /neo4j/plugins:/plugins \
#     -v /neo4j/conf:/conf \
#     --env NEO4J_PLUGINS='["apoc", "n10s"]' \
#     --env NEO4J_AUTH=neo4j/secretserver \
#     --env NEO4J_db_transaction_timeout=60s \
#     --env NEO4J_db_lock_acquisition_timeout=60s \
#     --env CLASSPATH_PREFIX=/neo4j/lib/dozerdb-plugin-5.20.0.0.jar \
#     graphstack/dozerdb:5.20.0.0-alpha.1

# # Populate databases
# python seq2seq/serve_rdf4j_graphs.py ~/git/balanced-plms/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split dev
# python seq2seq/serve_neo4j_graphs.py ~/git/balanced-plms/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split dev

# ______________________

# __ON SUBSEQUENT RUNS__
docker start rdf4j_server
docker start neo4j_server


python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/sql_compact.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/sparql_compact.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/cypher_compact.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/sql_norange.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/sparql_norange.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/cypher_norange.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/sql_no-schema.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/sparql_no-schema.json
python seq2seq/eval_vllm.py configs/3_eval_Llama3.1-8b/cypher_no-schema.json


