#!/bin/bash
# Ablation study: ECL - NO_HIPPO

export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
batch_size=${batch_size:-32}

# NO_HIPPO

# NO_HIPPO - Pred Len: 96
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'NO_HIPPO' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--train_epochs 5 \
	--itr 1 \
	--d_state 32 \
	--r_rank 8 \
	--ode_steps 10 \
	--ode_type 'dynamic' \
	--replace_block 'none' \
	--learning_rate 0.0004

# NO_HIPPO - Pred Len: 192
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_192 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 2 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'NO_HIPPO' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--train_epochs 5 \
	--itr 1 \
	--d_state 32 \
	--r_rank 8 \
	--ode_steps 10 \
	--ode_type 'dynamic' \
	--replace_block 'none' \
	--learning_rate 0.0005

# NO_HIPPO - Pred Len: 336
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_336 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers 2 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'NO_HIPPO' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--train_epochs 5 \
	--itr 1 \
	--d_state 16 \
	--r_rank 8 \
	--ode_steps 10 \
	--ode_type 'dynamic' \
	--replace_block 'none' \
	--learning_rate 0.0005

# NO_HIPPO - Pred Len: 720
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_720 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers 2 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'NO_HIPPO' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--train_epochs 5 \
	--itr 1 \
	--d_state 32 \
	--r_rank 16 \
	--ode_steps 10 \
	--ode_type 'dynamic' \
	--replace_block 'none' \
	--learning_rate 0.0005
