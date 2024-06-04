
export DEBUG=1


reference_models="microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct"

python generate_for_alpaca_eval.py \
    --model="Qwen/Qwen1.5-72B-Chat" \
    --output-path="outputs/Qwen-72B-round-1_MoA-Lite.json" \
    --reference-models=${reference_models} \
    --rounds 1 \
    --num-proc 1