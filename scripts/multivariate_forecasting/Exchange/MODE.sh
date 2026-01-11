export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
# Default e_layers is 2
e_layers=${e_layers:-3}
learning_rate=${learning_rate:-0.00005}
d_state=${d_state:-32}
r_rank=${r_rank:-16}
ode_steps=${ode_steps:-10}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/exchange_rate/ \
	--data_path exchange_rate.csv \
	--model_id Exchange_96_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 8 \
	--dec_in 8 \
	--c_out 8 \
	--des 'Exp' \
	--d_model 128 \
	--batch_size 16 \
	--d_ff 128 \
	--itr 1 \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/exchange_rate/ \
	--data_path exchange_rate.csv \
	--model_id Exchange_96_192 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers $e_layers \
	--enc_in 8 \
	--dec_in 8 \
	--c_out 8 \
	--des 'Exp' \
	--d_model 128 \
	--itr 1 \
	--d_ff 128 \
	--d_state $d_state \
	--r_rank 20 \
	--ode_steps $ode_steps \
	--learning_rate 0.000076


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/exchange_rate/ \
	--data_path exchange_rate.csv \
	--model_id Exchange_96_336 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers $e_layers \
	--enc_in 8 \
	--dec_in 8 \
	--c_out 8 \
	--des 'Exp' \
	--itr 1 \
	--d_model 128 \
	--d_ff 128 \
	--d_state 32 \
	--r_rank 24 \
	--ode_steps 10 \
	--learning_rate 0.000075

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/exchange_rate/ \
	--data_path exchange_rate.csv \
	--model_id Exchange_96_720 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers $e_layers \
	--enc_in 8 \
	--dec_in 8 \
	--c_out 8 \
	--des 'Exp' \
	--itr 1 \
	--d_model 128 \
	--d_ff 128 \
	--d_state $d_state \
	--r_rank 32 \
	--ode_steps $ode_steps \
	--learning_rate 0.000075
