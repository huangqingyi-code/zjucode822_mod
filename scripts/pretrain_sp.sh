# 记得根据卡的数量改grad acc以及accelerate_config
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH='/data0/pretrained-models'

export CUDA_VISIBLE_DEVICES=2,3,4,5
PER_DEVICES_BS=2
GRAD_ACC=8
ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_4.yaml

BASE_MODEL=/data4/sft_output/qwen2-base-0717/checkpoint-2000
REPORT_TO=wandb
PORT=20066

export WANDB__SERVICE_WAIT=600
# export WANDB_MODE=offline
export WANDB_MODE=online
NUM_EPOCHS=1

export MAX_COL=20
DATA_PATH=/data1/workspace/gxj/align_simple/train_20.json
EVAL_DATA_PATH=/data1/workspace/gxj/align_simple/test_20.json
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu3_zero3.yaml
HEAD_COUNT=1

LR=1e-4
EVAL_STEPS=250
DATA_COUNT=-1
LR_SCHE=constant_with_warmup

QNUM=10


ENCODER_PATH=/data0/ll/checkpoints/contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/model_25400.pt
# MODEL_NAME=align_${HEAD_COUNT}head_norm_1e-4
MODEL_NAME=${MAX_COL}col_${DATA_COUNT}_norm_prefer-not-num_align_contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717
EXP_NAME=sp_${LR}_${LR_SCHE}_notnorm
# EXP_NAME=warmup_${NUM_EPOCHS}epochs_${LR}_${LR_SCHE}_qnum${QNUM}_norm
# EXP_NAME=encdec_stage1_1epochs_1e-4_constant-with-warmup_freezeencoder
# ENCODER_PATH=checkpoints/utils/new_contrastive-semantic-11-None-0.0001_model_90.bin
# MODEL_NAME=new_contrastive

# ENCODER_PATH=checkpoints/utils/encoder_column_contrasive.bin
# MODEL_NAME=old_contrastive_norm
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_sp.py \
    --decoder_path ${BASE_MODEL} \
    --encoder_path $ENCODER_PATH \
    --projector_num_heads ${HEAD_COUNT} \
    --qformer_qnum ${QNUM} \
    --data_path $DATA_PATH \
    --data_count $DATA_COUNT \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir ./checkpoints/align/${MODEL_NAME}/${EXP_NAME} \
    --freeze_projector False \
    --freeze_encoder False \
    --freeze_decoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --evaluation_strategy "steps" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "steps" \
    --save_steps ${EVAL_STEPS} \
    --save_total_limit 30 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type ${LR_SCHE} \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained False \
    --torch_dtype float32
