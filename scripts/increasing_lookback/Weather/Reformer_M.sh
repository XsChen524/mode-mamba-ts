export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-Reformer_M}
batch_size=${batch_size:-16}

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
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_REFORM_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 5

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
	--e_layers 3 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_REFORM_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 5

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
	--e_layers 3 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_REFORM_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 5

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
	--e_layers 3 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_REFORM_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 5

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
	--e_layers 2 \
	--enc_in 21 \
	--dec_in 21 \
	--c_out 21 \
	--des 'LOOKBACK_REFORM_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size $batch_size \
	--learning_rate 0.00005 \
	--itr 1 \
	--train_epochs 5

