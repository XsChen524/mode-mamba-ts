export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.00005}
d_state=${d_state:-8}
r_rank=${r_rank:-4}
ode_steps=${ode_steps:-10}
ode_type=${ode_type:-'static'}

python -u scripts/run.py \
	--model_id ETTm1_48_96 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 48 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'LOOKBACK_MODE' \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--hippo \
	--learning_rate $learning_rate \

python -u scripts/run.py \
	--model_id ETTm1_96_96 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'LOOKBACK_MODE' \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--hippo \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--model_id ETTm1_192_96 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 192 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'LOOKBACK_MODE' \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model_id ETTm1_336_96 \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 336 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'LOOKBACK_MODE' \
	--d_model 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--hippo \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model_id ETTm1_720_96 \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 720 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'LOOKBACK_MODE' \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--ode_type $ode_type \
	--hippo \
	--learning_rate $learning_rate
