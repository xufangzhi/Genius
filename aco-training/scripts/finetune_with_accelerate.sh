source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/symbolv3
cd /cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct
echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d650b03778bb6cce1d21805bb757e4d42d222574
#MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873
#MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--EleutherAI--llemma_7b/snapshots/e223eee41c53449e6ea6548c9b71c50865e4a85c
# MODEL_DIR=/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct
# MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/symbol_llm_base_sft_tune_llama3.1chat_8B
# MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/4281e96c7cf5ab6b312ef0cb78373efa3976a9dd
#MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/afbda8b347ec881666061fa67447046fc5164ec8
MODEL_DIR=/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
# MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct/output/android_control_train_ac_full_llama3.1_8B


# MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct/output/241114-2_llama3.1_8B

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
    open_instruct/finetune.py \
    --model_name_or_path ${MODEL_DIR} \
    --use_flash_attn \
    --tokenizer_name ${MODEL_DIR} \
    --use_slow_tokenizer \
    --train_file ./data/241229-2.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./output/241229-2_llama3.1_${MODEL_SIZE} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1