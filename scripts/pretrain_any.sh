# NUM_EPOCHS=8
# DATA_PATH=/home/llong/gxj/DeepSeek-Coder/clean_new_pollute_spider_train_original.json
# ACCELERATE_CONFIG_PATH=/home/llong/gxj/DeepSeek-Coder/accelerate_config.yaml
# ENCODER_PATH=checkpoints/utils/encoder_name_prediction.bin
# MODEL_NAME=0629_nameguess

# NUM_EPOCHS=4
# DATA_PATH=/home/llong/gxj/DeepSeek-Coder/clean_new_pollute_spider_train_original.json
# ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config.yaml
# ENCODER_PATH=checkpoints/utils/new_contrastive-semantic-11-None-0.0001_model_90.bin
# MODEL_NAME=new_contrastive


# NUM_EPOCHS=4
# DATA_PATH=/home/llong/gxj/DeepSeek-Coder/clean_new_pollute_spider_train_original.json
# ACCELERATE_CONFIG_PATH=/home/llong/gxj/DeepSeek-Coder/accelerate_config.yaml
# ENCODER_PATH=checkpoints/utils/encoder_name_prediction.bin
# MODEL_NAME=0629_nameguess


# NUM_EPOCHS=4
# # DATA_PATH=/home/llong/gxj/DeepSeek-Coder/clean_new_pollute_spider_train_original.json
# ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config.yaml
# ENCODER_PATH=checkpoints/utils/encoder_column_contrasive.bin
# MODEL_NAME=test

NUM_EPOCHS=6
DATA_PATH=dataset/new/clean_comment_schema_new_pollute_spider_train_original.json
ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config.yaml
ENCODER_PATH=checkpoints/utils/encoder_name_prediction.bin
MODEL_NAME=insert_name_prediction


BASE_MODEL=/mnt/data/models/deepseek-ai/deepseek-coder-1.3b-instruct
REPORT_TO=wandb
PORT=20076
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_encoderdecoder.py \
    --base_model_name_or_path /mnt/data/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    --encoder_path $ENCODER_PATH \
    --data_path $DATA_PATH \
    --output_dir ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3 \
    --freeze_projector False \
    --freeze_encoder True \
    --freeze_decoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained False \
    --torch_dtype float32


accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_encoderdecoder.py \
    --base_model_name_or_path ${BASE_MODEL} \
    --encoder_path ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/encoder.bin \
    --decoder_path ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3 \
    --decoder_hidden_size 2048 \
    --data_path $DATA_PATH \
    --output_dir ./checkpoints/${MODEL_NAME}/encdec_stage2_${NUM_EPOCHS}epochs_1e-3 \
    --freeze_encoder False \
    --freeze_projector False \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained True \
    --n_tokens 0
    
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_encoderdecoder.py \
    --base_model_name_or_path ${BASE_MODEL} \
    --encoder_path ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/encoder.bin \
    --decoder_path ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3 \
    --decoder_hidden_size 2048 \
    --data_path $DATA_PATH \
    --output_dir ./checkpoints/${MODEL_NAME}/encdec_stage2_${NUM_EPOCHS}epochs_2e-4 \
    --freeze_encoder False \
    --freeze_projector False \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained True \
    --n_tokens 0


# accelerate launch \
#     --main_process_port ${PORT} \
#     --config_file ${ACCELERATE_CONFIG_PATH} \
#     train_encoderdecoder.py \
#     --base_model_name_or_path ${BASE_MODEL} \
#     --encoder_path ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/encoder.bin \
#     --decoder_path ./checkpoints/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3 \
#     --decoder_hidden_size 2048 \
#     --data_path $DATA_PATH \
#     --output_dir ./checkpoints/${MODEL_NAME}/encdec_stage2_${NUM_EPOCHS}+10epochs_2e-5_allunfreeze \
#     --freeze_encoder False \
#     --freeze_projector False \
#     --freeze_decoder False \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 3 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --load_pretrained True \
#     --n_tokens 0

# # todo: 交换次序的第二步
# accelerate launch \
#     --main_process_port 20277 \
#     train_encoderdecoder.py \
#     --base_model_name_or_path ${BASE_MODEL} \
#     --encoder_path ./checkpoints/new/encdec_spnprojfirst/encoder.bin \
#     --decoder_path ./checkpoints/new/encdec_spnprojfirst \
#     --learnable_embedding_path ./checkpoints/new/encdec_spnprojfirst/learnable_embedding.bin \
#     --learnable_embedding_projector_path ./checkpoints/new/encdec_spnprojfirst/learnable_embedding_projector.bin \
#     --data_path $DATA_PATH \
#     --output_dir ./checkpoints/new/encdec_spnprojfirst_nextall2e-4_8ep \
#     --freeze_encoder False \
#     --freeze_projector False \
#     --freeze_sp False \
#     --num_train_epochs 8 \
#     --n_tokens 10 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 3 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --load_pretrained True