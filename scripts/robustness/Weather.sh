#!/bin/bash

# Robustness testing for Weather dataset

export CUDA_VISIBLE_DEVICES=0

# Model and basic parameters
model_name=${MODEL_NAME:-MODE}
learning_rate=${learning_rate:-0.00006}
e_layers=${e_layers:-3}
d_state=${d_state:-8}
r_rank=${r_rank:-4}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}
noise_level=${noise_level:-0.1}

python -u scripts/run.py \
    --model_id weather_96_96 \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers $e_layers \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --train_epochs 5 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state 32 \
    --r_rank 16 \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.0001

python -u scripts/run.py \
    --model_id weather_96_192 \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers $e_layers \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --train_epochs 5 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state 8 \
    --r_rank 4 \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.000055

python -u scripts/run.py \
    --model_id weather_96_336 \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --train_epochs 5 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state 4 \
    --r_rank 2 \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.000065

python -u scripts/run.py \
    --model_id weather_96_720 \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --train_epochs 5 \
    --itr 1 \
    --noise_level $noise_level \
    --d_state 4 \
    --r_rank 2 \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.00006
