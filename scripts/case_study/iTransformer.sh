export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-iTransformer}

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
	--c_out 7 \
	--des 'CaseStudy_iTransformer' \
	--d_model 256 \
	--d_ff 256 \
	--learning_rate 0.00005 \
	--train_epochs 5 \
	--itr 1

python -u scripts/run.py \
	--model_id ETTm1_96_192 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 2 \
	--enc_in 7 \
	--c_out 7 \
	--des 'CaseStudy_iTransformer' \
	--d_model 256 \
	--d_ff 256 \
	--learning_rate 0.00005 \
	--train_epochs 5 \
	--itr 1

python -u scripts/run.py \
	--model_id ETTm1_96_384 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm1.csv \
	--model $model_name \
	--data ETTm1 \
	--features M \
	--seq_len 96 \
	--pred_len 384 \
	--e_layers 2 \
	--enc_in 7 \
	--c_out 7 \
	--des 'CaseStudy_iTransformer' \
	--d_model 256 \
	--d_ff 256 \
	--learning_rate 0.00005 \
	--train_epochs 5 \
	--itr 1
