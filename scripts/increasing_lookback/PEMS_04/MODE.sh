export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
e_layers=${e_layers:-4}
learning_rate=${learning_rate:-0.0005}
d_state=${d_state:-8}
r_rank=${r_rank:-4}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_48_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 48 \
	--pred_len 12 \
	--e_layers $e_layers \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_96_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 12 \
	--e_layers $e_layers \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_192_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 192 \
	--pred_len 12 \
	--e_layers $e_layers \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_336_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 336 \
	--pred_len 12 \
	--e_layers $e_layers \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_720_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 720 \
	--pred_len 12 \
	--e_layers $e_layers \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state 16 \
	--r_rank 8 \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--use_norm 0
