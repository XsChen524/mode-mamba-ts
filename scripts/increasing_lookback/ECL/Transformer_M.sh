export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-Transformer_M}

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
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_TRANS_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 1

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
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_TRANS_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 1

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
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_TRANS_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 1

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
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_TRANS_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 1

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
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_TRANS_M' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--learning_rate 0.0005 \
	--itr 1 \
	--train_epochs 1
