export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-S_Mamba}

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
	--des 'CaseStudy_S_Mamba' \
	--d_model 256 \
	--d_ff 256 \
	--d_state 2 \
	--learning_rate 0.00005 \
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
	--dec_in 7 \
	--c_out 7 \
	--des 'CaseStudy_S_Mamba' \
	--d_model 128 \
	--d_ff 128 \
	--d_state 2 \
	--learning_rate 0.00005 \
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
	--dec_in 7 \
	--c_out 7 \
	--des 'CaseStudy_S_Mamba' \
	--d_model 128 \
	--d_ff 128 \
	--d_state 2 \
	--learning_rate 0.00005 \
	--itr 1
