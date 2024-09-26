CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 20331 \
    train_sp.py \
    --base_model_name_or_path /mnt/data/models/deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --device cuda \
    --mm_projector_type mlp2x_gelu \
    --encoder_hidden_size 384 \
    --decoder_hidden_size 4096 \
    --n_tokens 10 \
    --data_path ./dataset/train_singletable.json \
    --torch_dtype bfloat16 \
    --output_dir ./checkpoints/10token_sp \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
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
    --report_to wandb
