export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.0002}
d_state=${d_state:-16}
r_rank=${r_rank:-8}
ode_steps=${ode_steps:-10}

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
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'Exp-MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--itr 1 \
	--use_norm 0 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_96_24 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 24 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'Exp-MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--itr 1 \
	--use_norm 0 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_96_48 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 48 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'Exp-MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--itr 1 \
	--use_norm 0 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model_id PEMS04_96_96 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'Exp-MODE' \
	--d_model 1024 \
	--d_ff 1024 \
	--itr 1 \
	--use_norm 0 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate
