export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.00035} # default 0.0001
e_layers=${e_layers:-2} # default 2
d_state=${d_state:-16} # default 16
r_rank=${r_rank:-8}
ode_steps=${ode_steps:-10}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/Solar/ \
	--data_path solar_AL.txt \
	--model_id solar_96_96 \
	--model $model_name \
	--data Solar \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 137 \
	--dec_in 137 \
	--c_out 137 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--d_state 32 \
	--r_rank 24 \
	--ode_steps $ode_steps \
	--learning_rate 0.00035

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/Solar/ \
	--data_path solar_AL.txt \
	--model_id solar_96_192 \
	--model $model_name \
	--data Solar \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 3 \
	--enc_in 137 \
	--dec_in 137 \
	--c_out 137 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.0003

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/Solar/ \
	--data_path solar_AL.txt \
	--model_id solar_96_336 \
	--model $model_name \
	--data Solar \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers $e_layers \
	--enc_in 137 \
	--dec_in 137 \
	--c_out 137 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.00035

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/Solar/ \
	--data_path solar_AL.txt \
	--model_id solar_96_720 \
	--model $model_name \
	--data Solar \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers $e_layers \
	--enc_in 137 \
	--dec_in 137 \
	--c_out 137 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.0003

