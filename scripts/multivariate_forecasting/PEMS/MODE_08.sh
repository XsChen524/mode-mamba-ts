export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.0015}
d_state=${d_state:-32}
r_rank=${r_rank:-16}
ode_steps=${ode_steps:-10}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS08.npz \
	--model_id PEMS08_96_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 12 \
	--e_layers 2 \
	--enc_in 170 \
	--dec_in 170 \
	--c_out 170 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--use_norm 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS08.npz \
	--model_id PEMS08_96_24 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 24 \
	--e_layers 2 \
	--enc_in 170 \
	--dec_in 170 \
	--c_out 170 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--use_norm 1 \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS08.npz \
	--model_id PEMS08_96_48 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 48 \
	--e_layers 4 \
	--enc_in 170 \
	--dec_in 170 \
	--c_out 170 \
	--des 'Exp-MODE' \
	--d_model 256 \
	--d_ff 256 \
	--batch_size 16 \
	--itr 1 \
	--use_norm 1 \
	--d_state $d_state \
	--r_rank 8 \
	--ode_steps $ode_steps \
	--learning_rate 0.0012

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS08.npz \
	--model_id PEMS08_96_96 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 170 \
	--dec_in 170 \
	--c_out 170 \
	--des 'Exp-MODE' \
	--d_model 256 \
	--d_ff 256 \
	--batch_size 16 \
	--itr 1 \
	--use_norm 1 \
	--d_state $d_state \
	--r_rank 8 \
	--ode_steps $ode_steps \
	--learning_rate 0.0012
