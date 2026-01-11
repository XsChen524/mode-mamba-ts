#!/bin/bash
# Ablation study: Weather - S_MAMBA

export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}

# S_MAMBA - Pred Len: 96
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
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'S_MAMBA' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 32 \
	--r_rank 16 \
	--expand 2 \
	--ode_type 'none' \
	--replace_block 's-mamba' \
	--learning_rate 0.0001

# S_MAMBA - Pred Len: 192
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
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'S_MAMBA' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 8 \
	--r_rank 4 \
	--expand 2 \
	--ode_type 'none' \
	--replace_block 's-mamba' \
	--learning_rate 0.000055

# S_MAMBA - Pred Len: 336
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
	--des 'S_MAMBA' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 4 \
	--r_rank 2 \
	--expand 2 \
	--ode_type 'none' \
	--replace_block 's-mamba' \
	--learning_rate 0.000065

# S_MAMBA - Pred Len: 720
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
	--des 'S_MAMBA' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--itr 1 \
	--d_state 4 \
	--r_rank 2 \
	--expand 2 \
	--ode_type 'none' \
	--replace_block 's-mamba' \
	--learning_rate 0.00006
