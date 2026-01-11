export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
e_layers=${e_layers:-3}
batch_size=${batch_size:-32}
learning_rate=${learning_rate:-0.0001}
d_state=${d_state:-8}
r_rank=${r_rank:-4}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}

# Using 8:4
python -u scripts/run.py \
	--model_id Weather_48_96 \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 48 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type

python -u scripts/run.py \
	--model_id Weather_96_96 \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--model_id Weather_192_96 \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 192 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--model_id Weather_336_96 \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 336 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type

python -u scripts/run.py \
	--model_id Weather_720_96 \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 720 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type
