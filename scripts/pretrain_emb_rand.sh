# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --main_process_port 20011 \
#     train_embedding_rand.py \
#     --base_model_name_or_path /home/xjgu/deepseek-coder-7b-instruct-v1.5 \
#     --device cuda \
#     --db_path /home/xjgu/spider/database_emb \
#     --mm_projector_type mlp2x_gelu \
#     --encoder_hidden_size 384 \
#     --decoder_hidden_size 4096 \
#     --data_path /home/xjgu/gxj/code/dataset/train_spider_must.json \
#     --bf16 True \
#     --output_dir ./checkpoints/projector_pretrain/table_embs_4epoch_mlp2x-gelu_rand_normalize \
#     --num_train_epochs 4 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \~
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --main_process_port 20031 \
#     train_embedding.py \
#     --base_model_name_or_path /home/xjgu/deepseek-coder-7b-instruct-v1.5 \
#     --device cuda \
#     --db_path /home/xjgu/spider/database_emb \
#     --mm_projector_type mlp2x_gelu \
#     --encoder_hidden_size 384 \
#     --decoder_hidden_size 4096 \
#     --data_path /home/xjgu/gxj/code/dataset/train_spider_must.json \
#     --bf16 True \
#     --output_dir ./checkpoints/projector_pretrain/table_embs_4epoch_mlp2x-gelu_just-concat_normalize_2 \
#     --num_train_epochs 4 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --main_process_port 20011 \
    train_embedding_rand.py \
    --base_model_name_or_path /home/xjgu/deepseek-coder-7b-instruct-v1.5 \
    --device cuda \
    --db_path /home/xjgu/spider/database_emb \
    --mm_projector_type mlp2x_gelu \
    --encoder_hidden_size 384 \
    --decoder_hidden_size 4096 \
    --data_path /home/xjgu/gxj/code/dataset/train_singletable.json \
    --bf16 True \
    --output_dir ./checkpoints/projector_pretrain/table_embs_4epoch_mlp2x-gelu_singletable_rand \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
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
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \

