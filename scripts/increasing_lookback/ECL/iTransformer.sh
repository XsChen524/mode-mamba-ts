export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-iTransformer}

python -u scripts/run.py \
	--model_id ECL_48_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path middle_electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 48 \
	--pred_len 96 \
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1

python -u scripts/run.py \
	--model_id ECL_96_96 \
	--is_training 1 \
	--root_path ./data/electricity/ \
	--data_path middle_electricity.csv \
	--model $model_name \
	--data custom \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 3 \
	--enc_in 321 \
	--dec_in 321 \
	--c_out 321 \
	--des 'LOOKBACK_ITRANS' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--itr 1

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
	--des 'LOOKBACK_ITRANS' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--train_epochs 5 \
	--learning_rate 0.0005 \
	--itr 1

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
	--des 'LOOKBACK_ITRANS' \
	--d_model 512 \
	--d_ff 512 \
	--batch_size 16 \
	--train_epochs 5 \
	--learning_rate 0.0005 \
	--itr 1

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
	--des 'LOOKBACK_ITRANS' \
	--d_model 512 \
	--d_ff 512 \
	--train_epochs 5 \
	--batch_size 16 \
	--learning_rate 0.0005 \
	--itr 1

