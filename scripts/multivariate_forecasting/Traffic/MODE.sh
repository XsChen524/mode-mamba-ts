export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.0005}
r_rank=${r_rank:-16}
d_state=${d_state:-32}
ode_steps=${ode_steps:-10}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model_id traffic_96_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model_id traffic_96_192 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model_id traffic_96_336 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model_id traffic_96_720 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--c_out 862 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1
