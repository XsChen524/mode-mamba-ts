#!/bin/bash
# Ablation study: Weather - STATIC_ODE

export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
# STATIC_ODE

# STATIC_ODE - Pred Len: 96
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_96_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 3 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'STATIC_ODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 32 \
	--r_rank 16 \
	--ode_steps 10 \
	--hippo \
	--ode_type 'static' \
	--replace_block 'none' \
	--learning_rate 0.0001

# STATIC_ODE - Pred Len: 192
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_96_192 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 3 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'STATIC_ODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 8 \
	--r_rank 4 \
	--ode_steps 10 \
	--hippo \
	--ode_type 'static' \
	--replace_block 'none' \
	--learning_rate 0.000055

# STATIC_ODE - Pred Len: 336
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_96_336 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'STATIC_ODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 4 \
	--r_rank 2 \
	--ode_steps 10 \
	--hippo \
	--ode_type 'static' \
	--replace_block 'none' \
	--learning_rate 0.000065

# STATIC_ODE - Pred Len: 720
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_96_720 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'STATIC_ODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 4 \
	--r_rank 2 \
	--ode_steps 10 \
	--hippo \
	--ode_type 'static' \
	--replace_block 'none' \
	--learning_rate 0.00006
