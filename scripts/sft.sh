# 记得根据卡的数量改grad acc以及accelerate_config
# 记得根据卡的数量改grad acc以及accelerate_config
# 记得根据卡的数量改grad acc以及accelerate_config
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NzgyOWMyOS01MDExLTQzNTMtYjZiZi1lYTNiNTAxMzMyNTAifQ=="
export NEPTUNE_PROJECT="gxj233/table-decoder"
export SENTENCE_TRANSFORMER='all-MiniLM-L6-v2'
export MODELS_PATH='/data0/pretrained-models'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
PER_DEVICES_BS=2
GRAD_ACC=128
let BS=PER_DEVICES_BS*GPU_COUNT*GRAD_ACC
echo "BS: $BS, GPU_COUNT: $GPU_COUNT"
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu8_zero3_acc64.yaml
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu4_zero3_acc128.yaml
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu6_zero3_acc5.yaml
# ACCELERATE_CONFIG_PATH=accelerate_config/5_acc8_zero2.yaml

ACCELERATE_CONFIG_PATH=accelerate_config/accelerate_config_${GPU_COUNT}_new.yaml
# ACCELERATE_CONFIG_PATH=accelerate_config/gpu3_zero3_acc171_gather.yaml
#ACCELERATE_CONFIG_PATH=accelerate_config/gpu4_zero3_acc128_gather.yaml

REPORT_TO=wandb
PORT=33445

export WANDB__SERVICE_WAIT=600
# export WANDB_MODE=offline
export WANDB_MODE=online
NUM_EPOCHS=1

# 注意统一align & sft的setting，老版本是(30, 20)，新版本是(20, 50)
export MAX_COL=20
export MAX_ROW=50

# Task: align with warmup + sft, old data format
DATA_PATH=/data1/workspace/gxj/sft_datasets/merge/table4_new/merge_sft_train.json
EVAL_DATA_PATH=/data1/workspace/gxj/sft_datasets/merge/table4_new/merge_sft_eval.json
# DATA_PATH=/data1/workspace/gxj/sft_datasets/merge/merge_sft_train.json
# EVAL_DATA_PATH=/data1/workspace/gxj/sft_datasets/merge/merge_sft_eval.json
HEAD_COUNT=1
LR=1e-5
LR_SCHE=constant_with_warmup
DATA_COUNT=-1

PRETRAINED_PATH=/data0/gxj/align/checkpoints_complex/mul_20col_-1/stage1_1epochs_1e-4_cosine_ratio0.01_qnum3_notnorm_1to1/checkpoint-57046
# PRETRAINED_PATH=/data0/gxj/sft_checkpoints/20col_-1/lr1e-5_constant_with_warmup_bs1032_bf16_freezedecoder/checkpoint-60
# PRETRAINED_PATH=/data0/gxj/sft_checkpoints/20col_-1/lr1e-5_constant_bs1024_bf16_freezedecoder_143840/checkpoint-100
# BASE_MODEL=/data4/sft_output/qwen2-base-0717/checkpoint-2000
# ENCODER_PATH=checkpoints/align/20col_-1_prefer-not-num_align_contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/encdec_stage1_1epochs_1e-3_freezeencoder/encoder_state_dict.bin
# PROJECTOR_PATH=checkpoints/align/20col_-1_prefer-not-num_align_contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/encdec_stage1_1epochs_1e-3_freezeencoder/projector.bin
# MODEL_NAME=${MAX_COL}col_${DATA_COUNT}
MODEL_NAME=20col_-1
EXP_NAME=lr${LR}_${LR_SCHE}_bs${BS}_bf16_onlyproj_table4_nods_new
EVAL_STEPS=40
SAVE_STEPS=20

# MODEL_NAME=old_contrastive_norm
accelerate launch \
    --main_process_port ${PORT} \
    --config_file ${ACCELERATE_CONFIG_PATH} \
    sft.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --projector_num_heads ${HEAD_COUNT} \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --data_count $DATA_COUNT \
    --output_dir /data1/workspace/gxj/sft_checkpoints/${MODEL_NAME}/${EXP_NAME} \
    --freeze_projector False \
    --freeze_encoder True \
    --freeze_decoder True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICES_BS} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --evaluation_strategy "steps" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 40 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type ${LR_SCHE} \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --load_pretrained True
