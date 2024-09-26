export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
PER_DEVICES_BS=2
GRAD_ACC=16
BASE_MODEL=/data0/workspace/liliyao/models/deepseek-ai/deepseek-coder-1.3b-instruct
REPORT_TO=wandb
PORT=20008
export WANDB__SERVICE_WAIT=600
export CUDA_VISIBLE_DEVICES=4,5
export WANDB_MODE=offline
NUM_EPOCHS=4
DATA_PATH=dataset/new/clean_comment_schema_new_pollute_spider_train_original.json
ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_2.yaml
ENCODER_PATH=checkpoints/utils/encoder_name_prediction.bin
HEAD_COUNT=16
MODEL_NAME=ins_namepred_${HEAD_COUNT}head_norm
# ENCODER_PATH=checkpoints/utils/new_contrastive-semantic-11-None-0.0001_model_90.bin
# MODEL_NAME=new_contrastive

ENCODER_PATH=checkpoints/utils/encoder_column_contrasive.bin
MODEL_NAME=old_contrastive_norm
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_ins_multihead.py \
    --decoder_path ${BASE_MODEL} \
    --encoder_path $ENCODER_PATH \
    --projector_num_heads ${HEAD_COUNT} \
    --data_path $DATA_PATH \
    --output_dir ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3 \
    --freeze_projector False \
    --freeze_encoder True \
    --freeze_decoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACC} \
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

# wait
# accelerate launch \
#     --main_process_port ${PORT} \
#     --config_file ${ACCELERATE_CONFIG_PATH} \
#     train_ins_multihead.py \
#     --decoder_path ${BASE_MODEL} \
#     --encoder_path ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/encoder.bin \
#     --projector_path ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/projector.bin \
#     --projector_num_heads ${HEAD_COUNT} \
#     --data_path $DATA_PATH \
#     --output_dir ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage2_${NUM_EPOCHS}epochs_1e-3 \
#     --freeze_projector False \
#     --freeze_encoder False \
#     --freeze_decoder True \
#     --num_train_epochs ${NUM_EPOCHS} \
#     --per_device_train_batch_size ${PER_DEVICES_BS} \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps ${GRAD_ACC} \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to ${REPORT_TO} \
#     --load_pretrained True \
#     --torch_dtype float32
# wait
# accelerate launch \
#     --main_process_port ${PORT} \
#     --config_file ${ACCELERATE_CONFIG_PATH} \
#     train_ins_multihead.py \
#     --decoder_path ${BASE_MODEL} \
#     --encoder_path ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/encoder.bin \
#     --projector_path ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/projector.bin \
#     --projector_num_heads ${HEAD_COUNT} \
#     --data_path $DATA_PATH \
#     --output_dir ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage2_${NUM_EPOCHS}epochs_2e-4 \
#     --freeze_projector False \
#     --freeze_encoder False \
#     --freeze_decoder True \
#     --num_train_epochs ${NUM_EPOCHS} \
#     --per_device_train_batch_size ${PER_DEVICES_BS} \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps ${GRAD_ACC} \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to ${REPORT_TO} \
#     --load_pretrained True \
#     --torch_dtype float32
# wait
# accelerate launch \
#     --main_process_port ${PORT} \
#     --config_file ${ACCELERATE_CONFIG_PATH} \
#     train_ins_multihead.py \
#     --decoder_path ${BASE_MODEL} \
#     --encoder_path ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/encoder.bin \
#     --projector_path ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage1_${NUM_EPOCHS}epochs_1e-3/projector.bin \
#     --projector_num_heads ${HEAD_COUNT} \
#     --data_path $DATA_PATH \
#     --output_dir ./checkpoints/ins_multihead/${MODEL_NAME}/encdec_stage2_${NUM_EPOCHS}+8epochs_2e-4 \
#     --freeze_projector False \
#     --freeze_encoder False \
#     --freeze_decoder True \
#     --num_train_epochs 8 \
#     --per_device_train_batch_size ${PER_DEVICES_BS} \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps ${GRAD_ACC} \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to ${REPORT_TO} \
#     --load_pretrained True \
#     --torch_dtype float32
# wait