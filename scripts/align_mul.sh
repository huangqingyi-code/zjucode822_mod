# 记得根据卡的数量改grad acc以及accelerate_config
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH='/data0/pretrained-models'

export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
PER_DEVICES_BS=2
GRAD_ACC=6

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu4_zero3_acc128_gather.yaml
ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_${GPU_COUNT}.yaml


REPORT_TO=wandb
PORT=38900

export WANDB__SERVICE_WAIT=600
export WANDB_MODE=online
NUM_EPOCHS=1

export MAX_COL=20
export MAX_ROW=50
DATA_PATH=/home/llong/gxj/code/dataset/sft_multitask/merge/merge_1to1.json
EVAL_DATA_PATH=/data1/workspace/gxj/align/complex/test_20.json
# ACCELERATE_CONFIG_PATH=aÏccelerate_config/gpu3_zero3.yaml
HEAD_COUNT=1

LR=1e-4
WARMUP_RATIO=0.01
EVAL_STEPS=125
SAVE_STEPS=500
DATA_COUNT=-1
LR_SCHE=cosine

QNUM=3


# BASE_MODEL=/data4/sft_output/qwen2-base-0802/checkpoint-2400
BASE_MODEL=/data0/pretrained-models/Qwen2-7B
ENCODER_PATH=/data0/ll/checkpoints/contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/model_25400.pt
# MODEL_NAME=align_${HEAD_COUNT}head_norm_1e-4
MODEL_NAME=mul_${MAX_COL}col_${DATA_COUNT}
NORM=not
EXP_NAME=stage1_${NUM_EPOCHS}epochs_${LR}_${LR_SCHE}_ratio${WARMUP_RATIO}_qnum${QNUM}_${NORM}norm_1to1_freezeencoder
OUTPUT_DIR=/data0/gxj/align/checkpoints_complex/${MODEL_NAME}/${EXP_NAME}
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    train_align.py \
    --decoder_path ${BASE_MODEL} \
    --encoder_path $ENCODER_PATH \
    --projector_num_heads ${HEAD_COUNT} \
    --qformer_qnum ${QNUM} \
    --data_path $DATA_PATH \
    --data_count $DATA_COUNT \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir ${OUTPUT_DIR} \
    --freeze_projector False \
    --freeze_encoder True \
    --freeze_decoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --evaluation_strategy "steps" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 88 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type ${LR_SCHE} \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained False \
    --torch_dtype float32
