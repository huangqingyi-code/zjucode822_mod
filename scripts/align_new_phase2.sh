# 记得根据卡的数量改grad acc以及accelerate_config
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH='/data0/pretrained-models'

export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
PER_DEVICES_BS=2
GRAD_ACC=6
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu4_zero3_acc128_gather.yaml
ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_${GPU_COUNT}.yaml


BASE_MODEL=/data4/sft_output/qwen2-base-0717/checkpoint-2000
REPORT_TO=wandb
PORT=20099

export WANDB__SERVICE_WAIT=600
export WANDB_MODE=online
NUM_EPOCHS=1

export MAX_COL=20
export MAX_ROW=50
DATA_PATH=/data1/workspace/gxj/align/complex/train_20.json
EVAL_DATA_PATH=/data1/workspace/gxj/align/complex/test_20.json
# ACCELERATE_CONFIG_PATH=aÏccelerate_config/gpu3_zero3.yaml
HEAD_COUNT=1

LR=5e-3
EVAL_STEPS=125
SAVE_STEPS=500
DATA_COUNT=-1
LR_SCHE=constant

QNUM=3

STAGE1_STEPS=500
MODEL_PATH=/data0/gxj/align/checkpoints_complex/20col_-1/stage1_1epochs_1e-3_cosine_ratio0.01_qnum3_notnorm_/checkpoint-${STAGE1_STEPS}
# MODEL_NAME=align_${HEAD_COUNT}head_norm_1e-4
MODEL_NAME=${MAX_COL}col_${DATA_COUNT}_notnorm

EXP_NAME=stage2from${STAGE1_STEPS}_${NUM_EPOCHS}epochs_${LR}_${LR_SCHE}_qnum${QNUM}_notnorm
OUTPUT_DIR=/data0/gxj/align/checkpoints_complex/${MODEL_NAME}/${EXP_NAME}
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
    --save_total_limit 88 \
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
