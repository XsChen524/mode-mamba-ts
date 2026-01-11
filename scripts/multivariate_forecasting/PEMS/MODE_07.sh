export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.0005}
r_rank=${r_rank:-16}
d_state=${d_state:-32}
ode_steps=${ode_steps:-10}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS07.npz \
	--model_id PEMS07_96_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 12 \
	--e_layers 2 \
	--enc_in 883 \
	--dec_in 883 \
	--c_out 883 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1 \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS07.npz \
	--model_id PEMS07_96_24 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 24 \
	--e_layers 2 \
	--enc_in 883 \
	--dec_in 883 \
	--c_out 883 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1 \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS07.npz \
	--model_id PEMS07_96_48 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 48 \
	--e_layers 4 \
	--enc_in 883 \
	--dec_in 883 \
	--c_out 883 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1 \
	--use_norm 0

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS07.npz \
	--model_id PEMS07_96_96 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 883 \
	--dec_in 883 \
	--c_out 883 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1 \
	--use_norm 0
