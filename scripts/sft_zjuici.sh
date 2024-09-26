export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH='/data0/pretrained-models'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
PER_DEVICES_BS=1
GRAD_ACC=64
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"

# ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_${GPU_COUNT}.yaml
ACCELERATE_CONFIG_PATH=accelerate_config/gpu2_zero3.yaml

REPORT_TO="wandb"
PORT=33445

export WANDB__SERVICE_WAIT=600
export WANDB_MODE=online


export MAX_COL=20
export MAX_ROW=50

# DATA_PATH=/data4/code822/tableqa/v18_full_20_train.json
DATA_PATH=/data4/code822/tableqa/mix.json
EVAL_DATA_PATH=/data4/code822/tableqa/v18_full_20_val.json
HEAD_COUNT=1
LR=1e-5
LR_SCHE=constant_with_warmup
DATA_COUNT=-1
NUM_EPOCHS=2
DATE=0913

PRETRAINED_PATH=/data0/gxj/align/checkpoints_complex/mul_20col_-1/0903_ckpt08171800_1epochs_5e-5_cosine_ratio0.03_qnum3_unfreeze_two/checkpoint-414
EXP_NAME=${DATE}_${LR}_${LR_SCHE}_bs${BS}
EVAL_STEPS=100
SAVE_STEPS=100

accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    sft.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --projector_num_heads ${HEAD_COUNT} \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --data_count $DATA_COUNT \
    --output_dir /data4/sft_checkpoints/${EXP_NAME} \
    --freeze_projector False \
    --freeze_encoder True \
    --freeze_decoder False \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --eval_strategy "steps" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 40 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.08 \
    --lr_scheduler_type ${LR_SCHE} \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --load_pretrained True