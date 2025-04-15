echo "[INFO] We have successfully activate the environment."
echo "[INFO] Start to run the shell." 

# you need 8 GPUs for full finetuning
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


MODEL_DIR=<path_to_model_dir>

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
    open_instruct/aco_tune.py \
    --model_name_or_path ${MODEL_DIR} \
    --use_flash_attn \
    --gradient_checkpointing \
    --tokenizer_name ${MODEL_DIR} \
    --use_slow_tokenizer \
    --train_file <path_to_data> \
    --max_seq_length 1300 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir <path_to_output_dir> \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --has_weights \