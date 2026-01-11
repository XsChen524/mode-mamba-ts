export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
learning_rate=${learning_rate:-0.00003}
e_layers=${e_layers:-2}
d_state=${d_state:-4}
r_rank=${r_rank:-2}
ode_steps=${ode_steps:-10}

# d state 2, MODE with r_rank=8
python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh2.csv \
	--model_id ETTh2_96_96 \
	--model $model_name \
	--data ETTh2 \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers $e_layers \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.00004

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh2.csv \
	--model_id ETTh2_96_192 \
	--model $model_name \
	--data ETTh2 \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers $e_layers \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.00007

python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh2.csv \
	--model_id ETTh2_96_336 \
	--model $model_name \
	--data ETTh2 \
	--features M \
	--seq_len 96 \
	--pred_len 336 \
	--e_layers $e_layers \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.000045


python -u scripts/run.py \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTh2.csv \
	--model_id ETTh2_96_720 \
	--model $model_name \
	--data ETTh2 \
	--features M \
	--seq_len 96 \
	--pred_len 720 \
	--e_layers $e_layers \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--des 'Exp' \
	--d_model 256 \
	--d_ff 256 \
	--itr 1 \
	--d_state $d_state \
	--r_rank $r_rank \
	--ode_steps $ode_steps \
	--learning_rate 0.000037
