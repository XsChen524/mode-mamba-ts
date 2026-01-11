export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-iTransformer}

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_48_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 48 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 256 \
	--d_ff 256 \
	--train_epochs 5 \
	--batch_size 16 \
	--learning_rate 0.0001 \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_96_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 256 \
	--d_ff 256 \
	--train_epochs 5 \
	--batch_size 16 \
	--learning_rate 0.0001 \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_192_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 192 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 256 \
	--d_ff 256 \
	--batch_size 16 \
	--train_epochs 5 \
	--learning_rate 0.0001 \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_336_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 336 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 256 \
	--d_ff 256 \
	--batch_size 16 \
	--train_epochs 5 \
	--learning_rate 0.0001 \
	--itr 1

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/weather/ \
	--data_path weather.csv \
	--model_id Weather_720_96 \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 720 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 256 \
	--d_ff 256 \
	--batch_size 16 \
	--train_epochs 5 \
	--learning_rate 0.0001 \
	--itr 1
