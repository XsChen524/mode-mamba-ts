#!/bin/bash
# Ablation study: ETTm1 - FULL-R-RANK

export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
ode_type=${ode_type:-'static'}

# FULL-R-RANK - Pred Len: 96
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model_id ETTm1_96_96 \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'FULL-R-RANK' \
	--d_ff 256 \
	--itr 1 \
	--d_state 8 \
	--r_rank 8 \
	--ode_steps 10 \
	--hippo \
	--ode_type $ode_type \
	--replace_block 'none' \
	--learning_rate 5e-05

# FULL-R-RANK - Pred Len: 192
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model_id ETTm1_96_192 \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'FULL-R-RANK' \
	--d_ff 128 \
	--itr 1 \
	--d_state 8 \
	--r_rank 8 \
	--ode_steps 10 \
	--hippo \
	--ode_type $ode_type \
	--replace_block 'none' \
	--learning_rate 5e-05

# FULL-R-RANK - Pred Len: 336
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model_id ETTm1_96_336 \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'FULL-R-RANK' \
	--d_model 128 \
	--itr 1 \
	--d_state 4 \
	--r_rank 4 \
	--ode_steps 10 \
	--hippo \
	--ode_type $ode_type \
	--replace_block 'none' \
	--learning_rate 7e-05

# FULL-R-RANK - Pred Len: 720
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model_id ETTm1_96_720 \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'FULL-R-RANK' \
	--d_ff 128 \
	--itr 1 \
	--d_state 16 \
	--r_rank 16 \
	--ode_steps 10 \
	--hippo \
	--ode_type $ode_type \
	--replace_block 'none' \
	--learning_rate 5.5e-05
