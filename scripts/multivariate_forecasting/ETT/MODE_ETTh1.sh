export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.00007}
d_state=${d_state:-4}
r_rank=${r_rank:-2}
ode_steps=${ode_steps:-10}

# d state 2, MODE with r_rank=8
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh1.csv \
	--model_id ETTh1_96_96 \
	--model MODE \
	--data ETTh1 \
	--features M \
	--train_epochs 100 \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp-MODE' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state 8 \
	--r_rank 4 \
	--ode_steps 12 \
	--learning_rate 0.000085

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh1.csv \
	--model_id ETTh1_96_192 \
	--model $model_name \
	--data ETTh1 \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp-MODE' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--ode_steps $ode_steps \
	--r_rank $r_rank \
	--learning_rate 0.0001

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh1.csv \
	--model_id ETTh1_96_336 \
	--model $model_name \
	--data ETTh1 \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp-MODE' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh1.csv \
	--model_id ETTh1_96_720 \
	--model $model_name \
	--data ETTh1 \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers 2 \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp-MODE' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate $learning_rate
