export CUDA_VISIBLE_DEVICES=2

model_name=${model_name:-S_Mamba}

python -u scripts/run.py \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 344 \
  --dec_in 344 \
  --c_out 344 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --exp_name partial_train \
  --itr 1


