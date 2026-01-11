#!/bin/bash
# MODE变量泛化能力实验脚本（ECL数据集）
# 仿照S_Mamba实验设计模式

export CUDA_VISIBLE_DEVICES=0

model_name=${model_name:-MODE}
d_state=${d_state:-32}
r_rank=${r_rank:-16}
ode_steps=${ode_steps:-10}

echo "=== MODE Variate Generalization (ECL Dataset) ==="
echo "Model: $model_name"
echo "d_state: $d_state, r_rank: $r_rank"
echo "=================================================\n"

# 训练时使用不同数量的变量，测试泛化能力
for enc_in in 64 128 192 256; do
    echo "Training with $enc_in variables (out of 321 total)"
    echo "----------------------------------------"

    python -u scripts/run.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_partial_${enc_in}_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in $enc_in \
        --dec_in $enc_in \
        --c_out $enc_in \
        --des "VarGen-train${enc_in}" \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --exp_name partial_train \
        --d_state $d_state \
        --r_rank $r_rank \
        --ode_steps $ode_steps \
        --learning_rate 0.0005 \
        --train_epochs 10 \
        --batch_size 16

done

echo "\n=== All variate generalization experiments completed! ==="
echo "Results saved to result_long_term_forecast.txt"
