export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.0001}
r_rank=${r_rank:-16}
d_state=${d_state:-32}
ode_steps=${ode_steps:-10}

# d_state = 32, MODE with r_rank=8
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS03.npz \
	--model_id PEMS03_96_12 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 12 \
	--e_layers 4 \
	--enc_in 358 \
	--dec_in 358 \
	--c_out 358 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--learning_rate $learning_rate \
	--train_epochs 5 \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS03.npz \
	--model_id PEMS03_96_24 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 24 \
	--e_layers 4 \
	--enc_in 358 \
	--dec_in 358 \
	--c_out 358 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS03.npz \
	--model_id PEMS03_96_48 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 48 \
	--e_layers 5 \
	--enc_in 358 \
	--dec_in 358 \
	--c_out 358 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS03.npz \
	--model_id PEMS03_96_96 \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 358 \
	--dec_in 358 \
	--c_out 358 \
	--des 'Exp-MODE' \
	--d_model 512 \
	--d_ff 512 \
	--learning_rate $learning_rate \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--itr 1
