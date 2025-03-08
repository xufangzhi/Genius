source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/flashattv2
cd symbol-llm-v2/open-instruct
echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf
#MODEL_DIR=/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct
#MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v8_dpo_iter4_dpo_tune_sft_iter3_7B
#MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/theoremqa_sft_iter1_sft_tune_sft_iter0_7B
MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
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
    --train_file ./data/theoremqa_v1_sft_iter10_pro.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./output/theoremqa_v1_sft_iter10_pro_sft_tune_llama2chat_${MODEL_SIZE} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1