if [ -z "$1" ]; then
  echo "Usage: $0 <num_train_epochs>"
  exit 1
fi

NUM_EPOCHS=$1

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 20353 \
    train_encoderdecoder.py \
    --base_model_name_or_path /mnt/data/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    --encoder_path /home/xjgu/gxj/code/model.bin \
    --data_path ./dataset/train_new.json \
    --output_dir ./checkpoints/new/encdec_stage1_${NUM_EPOCHS}epochs_2x8 \
    --freeze_projector False \
    --freeze_encoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --load_pretrained False


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 20053 \
    train_encoderdecoder.py \
    --base_model_name_or_path /mnt/data/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    --encoder_path ./checkpoints/new/encdec_stage1_${NUM_EPOCHS}epochs_2x8/encoder.bin \
    --decoder_path ./checkpoints/new/encdec_stage1_${NUM_EPOCHS}epochs_2x8 \
    --data_path ./dataset/train_new.json \
    --output_dir ./checkpoints/new/encdec_stage2_${NUM_EPOCHS}epochs_2x8 \
    --freeze_encoder False \
    --freeze_projector False \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --load_pretrained True

