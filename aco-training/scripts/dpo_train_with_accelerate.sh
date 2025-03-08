source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/symbolv3
cd /cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct
echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell." 
# sleep 1h
# you need 8 GPUs for full finetuning
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1

MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/38143357bf52ce57009ecbd58cf9f0b0029cb393/
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de/
 
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/
# TOKENIZER_DIR=/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5/
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819/
# MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct/output/241217-1_llama3.1_8B
# MODEL_DIR=/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/241219-2_llama3.1_8B/

MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/dpo_tune.py \
    --model_name_or_path ${MODEL_DIR} \
    --use_flash_attn \
    --gradient_checkpointing \
    --tokenizer_name ${MODEL_DIR} \
    --use_slow_tokenizer \
    --train_file ./data/250126-3.jsonl \
    --max_seq_length 1300 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir /nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/250126-3_llama3.1_${MODEL_SIZE} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --has_weights \