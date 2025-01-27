# 记得根据卡的数量改grad acc以及accelerate_config
# todo: from_pretrained不支持改config，感觉还需要改改

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH='/data0/pretrained-models'

export CUDA_VISIBLE_DEVICES=5
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
PER_DEVICES_BS=4
GRAD_ACC=16
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu4_zero3_acc128_gather.yaml
ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_${GPU_COUNT}.yaml


REPORT_TO=wandb
PORT=20077

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
EVAL_STEPS=125
SAVE_STEPS=1000
DATA_COUNT=-1
LR_SCHE=cosine

QNUM=10

MODEL_PATH=checkpoints/align/20col_-1_norm_prefer-not-num_align_contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/warmup_1epochs_1e-4_constant_with_warmup_qnum10_notnorm/tmp_ckpt/checkpoint-500
MODEL_NAME=${MAX_COL}col_${DATA_COUNT}_notnorm_prefer-not-num_align_contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717
EXP_NAME=stage2_${NUM_EPOCHS}epochs_${LR}_${LR_SCHE}_bs${BS}_qnum${QNUM}_notnorm
# OUTPUT_DIR=/data1/workspace/gxj/align/checkpoints/${MODEL_NAME}/${EXP_NAME}
OUTPUT_DIR=/data0/gxj/align/checkpoints/${MODEL_NAME}/${EXP_NAME}
# ENCODER_PATH=checkpoints/utils/new_contrastive-semantic-11-None-0.0001_model_90.bin
# MODEL_NAME=new_contrastive

# ENCODER_PATH=checkpoints/utils/encoder_column_contrasive.bin
# MODEL_NAME=old_contrastive_norm
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_align.py \
    --pretrained_path ${MODEL_PATH} \
    --projector_num_heads ${HEAD_COUNT} \
    --qformer_qnum ${QNUM} \
    --data_path $DATA_PATH \
    --data_count $DATA_COUNT \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir ${OUTPUT_DIR} \
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
    --save_steps ${SAVE_STEPS} \
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
    --load_pretrained True \
    --torch_dtype float32
