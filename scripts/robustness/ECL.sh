#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Model and basic parameters
model_name=${MODEL_NAME:-MODE}
e_layers=${e_layers:-2}
batch_size=${batch_size:-32}
learning_rate=${learning_rate:-0.0005}
d_state=${d_state:-32}
r_rank=${r_rank:-8}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}
noise_level=${noise_level:-0.1}

python -u scripts/run.py \
    --model_id ECL_96_96 \
    --is_training 1 \
    --root_path ./data/electricity/ \
    --data_path electricity.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers $e_layers \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size $batch_size \
    --noise_level $noise_level \
    --d_state $d_state \
    --r_rank $r_rank \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate 0.0004

python -u scripts/run.py \
    --model_id ECL_96_192 \
    --is_training 1 \
    --root_path ./data/electricity/ \
    --data_path electricity.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers $e_layers \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size $batch_size \
    --noise_level $noise_level \
    --d_state $d_state \
    --r_rank $r_rank \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate $learning_rate

python -u scripts/run.py \
    --model_id ECL_96_336 \
    --is_training 1 \
    --root_path ./data/electricity/ \
    --data_path electricity.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers $e_layers \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size $batch_size \
    --noise_level $noise_level \
    --d_state 16 \
    --r_rank $r_rank \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate $learning_rate

python -u scripts/run.py \
    --model_id ECL_96_720 \
    --is_training 1 \
    --root_path ./data/electricity/ \
    --data_path electricity.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers $e_layers \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Robust' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size $batch_size \
    --noise_level $noise_level \
    --d_state $d_state \
    --r_rank 16 \
    --ode_steps $ode_steps \
    --ode_type $ode_type \
    --learning_rate $learning_rate
