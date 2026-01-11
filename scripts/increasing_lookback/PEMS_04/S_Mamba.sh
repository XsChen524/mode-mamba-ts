export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-S_Mamba}

python -u scripts/run.py \
	--model_id PEMS04_48_12 \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 48 \
	--pred_len 12 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_SMB' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate 0.0005 \
	--use_norm 0 \
	--itr 1

python -u scripts/run.py \
	--model_id PEMS04_96_12 \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 96 \
	--pred_len 12 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_SMB' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate 0.0005 \
	--use_norm 0 \
	--itr 1

python -u scripts/run.py \
	--model_id PEMS04_192_12 \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 192 \
	--pred_len 12 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_SMB' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate 0.0005 \
	--use_norm 0 \
	--itr 1

python -u scripts/run.py \
	--model_id PEMS04_336_12 \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 336 \
	--pred_len 12 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_SMB' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate 0.0005 \
	--use_norm 0 \
	--itr 1

python -u scripts/run.py \
	--model_id PEMS04_720_12 \
	--is_training 1 \
	--root_path ./data/PEMS/ \
	--data_path PEMS04.npz \
	--model $model_name \
	--data PEMS \
	--features M \
	--seq_len 720 \
	--pred_len 12 \
	--e_layers 4 \
	--enc_in 307 \
	--dec_in 307 \
	--c_out 307 \
	--des 'LOOKBACK_SMB' \
	--d_model 1024 \
	--d_ff 1024 \
	--learning_rate 0.0005 \
	--use_norm 0 \
	--itr 1
