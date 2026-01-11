export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-S_Mamba}

python -u scripts/run.py \
	--model_id traffic_48_96 \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 48 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'LOOKBACK_SMB' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1

python -u scripts/run.py \
	--model_id traffic_96_96 \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'LOOKBACK_SMB' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1

python -u scripts/run.py \
	--model_id traffic_192_96 \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 192 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'LOOKBACK_SMB' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1

python -u scripts/run.py \
	--model_id traffic_336_96 \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 336 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'LOOKBACK_SMB' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1

python -u scripts/run.py \
	--model_id traffic_720_96 \
	--is_training 1 \
	--root_path ./data/traffic/ \
	--data_path traffic.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 720 \
	--pred_len 96 \
	--e_layers 4 \
	--enc_in 862 \
	--dec_in 862 \
	--c_out 862 \
	--des 'LOOKBACK_SMB' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1
