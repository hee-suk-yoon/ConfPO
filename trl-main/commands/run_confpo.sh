#!/bin/bash
export TZ="Asia/Seoul"

OUTPUT_DIR=$1

MODEL_NAME="zephyr-mistral-base-sft"
MODEL_PATH="[ADD MODEL PATH HERE]"
# e.g. ~/workspace/models/zephyr-7b-sft-full

DATASET_NAME="HuggingFaceH4/ultrafeedback_binarized"
LOG_HOME="[ADD LOGGING DIR HERE]"

if [ "$OUTPUT_DIR" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT_DIR=${LOG_HOME}"/log-confpo/confpo-${MODEL_NAME/'/'/_}-$TIME_STEP"
fi
echo "OUTPUT_DIR: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR


# Handle extra arguments in case one passes accelerate configs.
EXTRA_ACCELERATE_ARGS=""
# """
EXTRA_TRAINING_ARGS=""

# Set your number of GPUs here
NUM_GPUS=4
TRL_ACCELERATE_CONFIG='./examples/accelerate_configs/deepspeed_zero2.yaml'
if [[ "${TRL_ACCELERATE_CONFIG}" == "" ]]; then
  EXTRA_ACCELERATE_ARGS=""
else
  EXTRA_ACCELERATE_ARGS="--config_file $TRL_ACCELERATE_CONFIG"
fi

RUN_NAME='confpo-zephyr-mistral-base-sft'

LR=3.0e-7
EPOCHS=1
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
PROMPT_LEN=1800
SEQ_LEN=2048
LOG_STEPS=10
EVAL_STEPS=50
SAVE_STEPS=100
DATASET_PROC=1
GRAD_ACCUM=4

BETA=1.5
GAMMA=1.6


CMD="""
accelerate launch $EXTRA_ACCELERATE_ARGS \
    --num_processes $NUM_GPUS \
    `pwd`/examples/scripts/cpo.py \
    --model_name $MODEL_PATH \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --dataset_train_split "train_prefs" \
    --dataset_test_split "test_prefs" \
    --max_length $SEQ_LEN \
    --max_prompt_length $PROMPT_LEN \
    --num_train_epochs $EPOCHS \
    --dataset_num_proc $DATASET_PROC \
    --run_name $RUN_NAME \
    --learning_rate $LR \
    --logging_steps $LOG_STEPS \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --bf16 \
    --do_eval \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --beta $BETA \
    --simpo_gamma $GAMMA \
    --lr_scheduler_type cosine \
    --chat_template "zephyr-mistral-base" \
    --loss_type simpo \
    --is_confpo \

    $EXTRA_TRAINING_ARGS
"""

echo "Starting program..."

{ # try
    echo $CMD
    # print log to termimal
    # eval "$CMD"
    
    # save log to file
    eval "$CMD" >> $OUTPUT_DIR/training.log 2>&1
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}
exit 0