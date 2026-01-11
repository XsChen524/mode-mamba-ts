export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
e_layers=${e_layers:-2}
batch_size=${batch_size:-32}
learning_rate=${learning_rate:-0.0005}
d_state=${d_state:-32}
r_rank=${r_rank:-8}
ode_steps=${ode_steps:-10}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.0004


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_192 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--train_epochs 5 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.0005

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_336 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--batch_size $batch_size \
	--train_epochs 5 \
	--d_state 16 \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.0005


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path electricity.csv \
	--model_id ECL_96_720 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers $e_layers \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'Exp' \
	--d_model 512 \
	--d_ff 512 \
	--itr 1 \
	--train_epochs 5 \
	--batch_size $batch_size \
	--d_state $d_state \
	--r_rank 16 \
	--ode_steps $ode_steps \
	--learning_rate 0.0005

