#!/bin/bash
# Individual efficiency experiment for a specific model and dataset
# This script now accepts command line parameters
# Usage: ./run_efficiency_exp.sh --model MODE --dataset ETTm1 --d_state 16 --r_rank 8 --ode_steps 10


# Parse command line arguments
MODEL_NAME="S_Mamba"  # Default model
DATASET_NAME="ETTm1"  # Default dataset

# Mamba model parameters (shared by all Mamba-based models: S_Mamba, MODE)
D_STATE=""

# MODE-specific parameters
R_RANK=""  # Low-rank parameter specific to MODE
D_CONV=""  # Convolution kernel size (fixed in S_Mamba but configurable in MODE)
EXPAND=""  # Expansion factor (fixed in S_Mamba but configurable in MODE)
ODE_STEPS=""  # Number of ODE steps for MODE (default: 10)
USE_NORM=""  # Use normalization flag (default: 1)

# BiMamba4TS parameters
E_FACT=""
BI_DIR=""
RESIDUAL=""
CH_IND=""
EMBED_TYPE=""  # Used by BiMamba4TS and Crossformer

# Patch-based parameters (used by PatchTST, BiMamba4TS, Crossformer)
PATCH_LEN=""
STRIDE=""
PADDING_PATCH=""  # Options: 'end' or other

# DLinear parameters
INDIVIDUAL=""  # Flag for DLinear model (1 if set)

# General training parameters
LEARNING_RATE=""

# PatchTST-specific parameters
FC_DROPOUT=""
HEAD_DROPOUT=""
PCT_START=""
LRADJ=""
DROPOUT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --d_state)
            D_STATE="$2"
            shift 2
            ;;
        --r_rank)
            R_RANK="$2"
            shift 2
            ;;
        --d_conv)
            D_CONV="$2"
            shift 2
            ;;
        --expand)
            EXPAND="$2"
            shift 2
            ;;
		--ode_steps)
            ODE_STEPS="$2"
            shift 2
            ;;
        --e_fact)
            E_FACT="$2"
            shift 2
            ;;
        --bi_dir)
            BI_DIR="$2"
            shift 2
            ;;
        --residual)
            RESIDUAL="$2"
            shift 2
            ;;
        --ch_ind)
            CH_IND="$2"
            shift 2
            ;;
        --embed_type)
            EMBED_TYPE="$2"
            shift 2
            ;;
        --patch_len)
            PATCH_LEN="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --padding_patch)
            PADDING_PATCH="$2"
            shift 2
            ;;
        --individual)
            INDIVIDUAL=1
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --use_norm)
            USE_NORM="$2"
            shift 2
            ;;
        --fc_dropout)
            FC_DROPOUT="$2"
            shift 2
            ;;
        --head_dropout)
            HEAD_DROPOUT="$2"
            shift 2
            ;;
        --pct_start)
            PCT_START="$2"
            shift 2
            ;;
        --lradj)
            LRADJ="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Dataset configuration
# S_Mamba experiments set the batch_size to 16.
if [ "$DATASET_NAME" = "ETTm1" ]; then
    ROOT_PATH="./data/ETT-small/"
    DATA_PATH="ETTm1.csv"
    DATA="ETTm1"
    ENC_IN=7
    DEC_IN=7
    C_OUT=7
    SEQ_LEN=96
    PRED_LEN=96
	E_LAYERS=2
	D_MODEL=256
	D_FF=256
	TRAIN_EPOCHS=10

elif [ "$DATASET_NAME" = "ECL" ]; then
    ROOT_PATH="./data/electricity/"
    DATA_PATH="electricity.csv"
    DATA="custom"
    ENC_IN=321
    DEC_IN=321
    C_OUT=321
    SEQ_LEN=96
    PRED_LEN=336
	E_LAYERS=3
	D_MODEL=512
	D_FF=512
	TRAIN_EPOCHS=5

elif [ "$DATASET_NAME" = "Weather" ]; then
    ROOT_PATH="./data/weather/"
    DATA_PATH="weather.csv"
    DATA="custom"
    ENC_IN=21
    DEC_IN=21
    C_OUT=21
    SEQ_LEN=96
    PRED_LEN=96
	E_LAYERS=3
	D_MODEL=512
	D_FF=512
	TRAIN_EPOCHS=5


elif [ "$DATASET_NAME" = "PEMS08" ]; then
    ROOT_PATH="./data/PEMS/"
    DATA_PATH="PEMS08.npz"
    DATA="PEMS"
    ENC_IN=170
    DEC_IN=170
    C_OUT=170
	E_LAYERS=2
    SEQ_LEN=96
    PRED_LEN=12
	D_MODEL=512
	D_FF=512

	TRAIN_EPOCHS=10

else
    echo "Error: Unknown dataset $DATASET_NAME"
    exit 1
fi

# Common config
TRAIN_EPOCHS=10
BATCH_SIZE=16
N_HEADS=8
PATIENCE=3

# Set learning rate (use provided value or default)
if [ -n "$LEARNING_RATE" ]; then
    LR_VALUE=$LEARNING_RATE
else
    LR_VALUE=0.0001
fi

# Set use_norm (use provided value or default 1)
if [ -n "$USE_NORM" ]; then
    USE_NORM_VALUE=$USE_NORM
else
    USE_NORM_VALUE=1
fi

# Create model_id in format: DATASET_SEQ_LEN_PRED_LEN (e.g., ETTm1_96_96)
MODEL_ID="${DATASET_NAME}_${SEQ_LEN}_${PRED_LEN}"

# Output directory
OUTPUT_DIR="./output/efficiency_exp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${OUTPUT_DIR}/efficiency_results_${DATASET_NAME}.txt"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Parameters: Batch ${BATCH_SIZE}, Epochs ${TRAIN_EPOCHS}, LR ${LR_VALUE}"
echo "Model ID: ${MODEL_ID}"
echo "================================================================================"
echo ""

# Build the python command with optional MODE parameters
PYTHON_CMD="python -u scripts/run.py \
    --is_training 1 \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model_id \"${MODEL_ID}\" \
    --model $MODEL_NAME \
    --data $DATA \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers $E_LAYERS \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --des 'Efficiency-Exp' \
    --batch_size $BATCH_SIZE \
    --train_epochs $TRAIN_EPOCHS \
    --learning_rate $LR_VALUE \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --n_heads $N_HEADS \
    --itr 1 \
    --patience $PATIENCE \
    --use_norm $USE_NORM_VALUE"

# Add model-specific parameters if provided
# d_state is shared by all Mamba-based models
if [ -n "$D_STATE" ]; then
    PYTHON_CMD="$PYTHON_CMD --d_state $D_STATE"
    echo "d_state: $D_STATE"
fi

# MODE-specific parameters
if [ "$MODEL_NAME" = "MODE" ]; then
    if [ -n "$R_RANK" ]; then
        PYTHON_CMD="$PYTHON_CMD --r_rank $R_RANK"
        echo "MODE r_rank: $R_RANK"
    fi

    if [ -n "$D_CONV" ]; then
        PYTHON_CMD="$PYTHON_CMD --d_conv $D_CONV"
        echo "MODE d_conv: $D_CONV"
    fi

    if [ -n "$EXPAND" ]; then
        PYTHON_CMD="$PYTHON_CMD --expand $EXPAND"
        echo "MODE expand: $EXPAND"
    fi

    # Set default ode_steps to 10 if not provided
    if [ -z "$ODE_STEPS" ]; then
        ODE_STEPS=10
    fi
    PYTHON_CMD="$PYTHON_CMD --ode_steps $ODE_STEPS"
    echo "MODE ode_steps: $ODE_STEPS"
fi

# BiMamba4TS-specific parameters
if [ "$MODEL_NAME" = "BiMamba4TS" ]; then
    if [ -n "$E_FACT" ]; then
        PYTHON_CMD="$PYTHON_CMD --e_fact $E_FACT"
        echo "BiMamba4TS e_fact: $E_FACT"
    fi

    if [ -n "$BI_DIR" ]; then
        PYTHON_CMD="$PYTHON_CMD --bi_dir $BI_DIR"
        echo "BiMamba4TS bi_dir: $BI_DIR"
    fi

    if [ -n "$RESIDUAL" ]; then
        PYTHON_CMD="$PYTHON_CMD --residual $RESIDUAL"
        echo "BiMamba4TS residual: $RESIDUAL"
    fi

    if [ -n "$CH_IND" ]; then
        PYTHON_CMD="$PYTHON_CMD --ch_ind $CH_IND"
        echo "BiMamba4TS ch_ind: $CH_IND"
    fi

    if [ -n "$EMBED_TYPE" ]; then
        PYTHON_CMD="$PYTHON_CMD --embed_type $EMBED_TYPE"
        echo "BiMamba4TS embed_type: $EMBED_TYPE"
    fi
fi

# DLinear-specific parameter (flag)
if [ -n "$INDIVIDUAL" ]; then
    PYTHON_CMD="$PYTHON_CMD --individual"
    echo "DLinear individual: true"
fi

# Patch-based parameters (used by PatchTST, BiMamba4TS, Crossformer)
if [ -n "$PATCH_LEN" ]; then
    PYTHON_CMD="$PYTHON_CMD --patch_len $PATCH_LEN"
    echo "patch_len: $PATCH_LEN"
fi

if [ -n "$STRIDE" ]; then
    PYTHON_CMD="$PYTHON_CMD --stride $STRIDE"
    echo "stride: $STRIDE"
fi

if [ -n "$PADDING_PATCH" ]; then
    PYTHON_CMD="$PYTHON_CMD --padding_patch $PADDING_PATCH"
    echo "padding_patch: $PADDING_PATCH"
fi

# PatchTST-specific parameters
if [ -n "$FC_DROPOUT" ]; then
    PYTHON_CMD="$PYTHON_CMD --fc_dropout $FC_DROPOUT"
    echo "fc_dropout: $FC_DROPOUT"
fi

if [ -n "$HEAD_DROPOUT" ]; then
    PYTHON_CMD="$PYTHON_CMD --head_dropout $HEAD_DROPOUT"
    echo "head_dropout: $HEAD_DROPOUT"
fi

if [ -n "$PCT_START" ]; then
    PYTHON_CMD="$PYTHON_CMD --pct_start $PCT_START"
    echo "pct_start: $PCT_START"
fi

if [ -n "$LRADJ" ]; then
    PYTHON_CMD="$PYTHON_CMD --lradj $LRADJ"
    echo "lradj: $LRADJ"
fi

if [ -n "$DROPOUT" ]; then
    PYTHON_CMD="$PYTHON_CMD --dropout $DROPOUT"
    echo "dropout: $DROPOUT"
fi

echo ""
echo "Starting training..."
echo ""

# Run training experiment and capture output
train_output=$(eval $PYTHON_CMD 2>&1)
train_exit_code=$?

# Check if training was successful
if [ $train_exit_code -ne 0 ]; then
    echo "ERROR: Training failed for ${MODEL_NAME} on ${DATASET_NAME}"
    echo ""
    echo "Error output:"
    echo "$train_output"
    exit 1
fi

echo ""
echo "Parsing metrics..."

# Pass the training output to the parse_efficiency_metrics.py script
# through stdin and pass other parameters as command line arguments
python scripts/efficiency/parse_efficiency_metrics.py \
    "${RESULTS_FILE}" \
    "${MODEL_NAME}" \
    "${DATASET_NAME}" \
    "${TIMESTAMP}" \
    "${MODEL_ID}" \
    --batch_size ${BATCH_SIZE} \
    --train_epochs ${TRAIN_EPOCHS} \
    --learning_rate ${LR_VALUE} \
    --d_model ${D_MODEL} \
    --e_layers ${E_LAYERS} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    < <(echo "${train_output}")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to parse metrics"
    exit 1
fi

echo "================================================================================"
echo "Results saved to: ${RESULTS_FILE}"
echo "================================================================================"
