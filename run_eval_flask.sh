
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

export DEBUG=0

mkdir -p outputs/flask

python generate_for_flask.py \
    --model="Qwen/Qwen1.5-72B-Chat" \
    --output-path="outputs/flask/Qwen1.5-72B-Chat-round-1.jsonl" \
    --reference-models="microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
    --rounds 1 \
    --num-proc 32

cd FLASK/gpt_review

python gpt4_eval.py \
    -a '../../outputs/flask/Qwen1.5-72B-Chat-round-1.jsonl' \
    -o '../../outputs/flask/chatgpt_review.jsonl'

python aggregate_skill.py -m '../../outputs/flask/chatgpt_review.jsonl'

cat outputs/stats/chatgpt_review_skill.csv