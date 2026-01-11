#!/bin/bash

# Robustness testing for ETTm1 dataset
# Single experiment with specified noise_level

export CUDA_VISIBLE_DEVICES=0

# Model and basic parameters
model_name=${MODEL_NAME:-MODE}
learning_rate=${learning_rate:-0.00005}
d_state=${d_state:-8}
r_rank=${r_rank:-4}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}
noise_level=${noise_level:-0.1}

python -u scripts/run.py \
    --model_id ETTm1_96_96 \
    --is_training 1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTm1.csv \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Robust' \
    --d_model 256 \
    --d_ff 256 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state $d_state \
    --r_rank $r_rank \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate $learning_rate

python -u scripts/run.py \
    --model_id ETTm1_96_192 \
    --is_training 1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTm1.csv \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Robust' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state $d_state \
    --r_rank $r_rank \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate $learning_rate

python -u scripts/run.py \
    --model_id ETTm1_96_336 \
    --is_training 1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTm1.csv \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Robust' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state 4 \
    --r_rank 2 \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.00007

python -u scripts/run.py \
    --model_id ETTm1_96_720 \
    --is_training 1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTm1.csv \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Robust' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state $d_state \
    --r_rank $r_rank \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.000055
