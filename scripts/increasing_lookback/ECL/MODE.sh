export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
e_layers=${e_layers:-2}
batch_size=${batch_size:-32}
learning_rate=${learning_rate:-0.0005}
d_state=${d_state:-32}
r_rank=${r_rank:-8}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}

# Using 32:8
python -u scripts/run.py \
	--model_id ECL_48_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 48 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--itr 1 \
	--hippo \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type

python -u scripts/run.py \
	--model_id ECL_96_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--itr 1 \
	--hippo \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--model_id ECL_192_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 192 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--train_epochs 5 \
	--hippo \
	--batch_size $batch_size \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--model_id ECL_336_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 336 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--itr 1 \
	--hippo \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type

python -u scripts/run.py \
	--model_id ECL_720_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 720 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_MODE' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--itr 1 \
	--hippo \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type
