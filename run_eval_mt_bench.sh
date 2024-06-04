export DEBUG=1


python generate_for_mt_bench.py --model "Qwen/Qwen1.5-72B-Chat" \
    --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
    --answer-file outputs/mt_bench/mt_bench-together-MoA-round1.jsonl \
    --parallel 1 --rounds 1

python eval_mt_bench.py --model-list mt_bench-together-MoA-round1 --parallel 32

python show_mt_bench_result.py