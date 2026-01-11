#!/bin/bash
# Efficiency experiments for ETTm1 dataset

# MODE on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model MODE \
	--dataset ETTm1 \
	--d_state 4 \
	--r_rank 1 \
	--d_conv 2 \
	--expand 2 \
	--ode_steps 2 \
	--learning_rate 0.00005

# PatchTST on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model PatchTST \
	--dataset ETTm1 \
	--patch_len 16 \
	--stride 8 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.0001
	# --lradj 'TST' \

# S_Mamba on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model S_Mamba \
	--dataset ETTm1 \
	--d_state 2 \
	--learning_rate 0.00005

# BiMamba4TS on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model BiMamba4TS \
	--dataset ETTm1 \
	--d_state 2 \
	--d_conv 2 \
	--e_fact 2 \
	--bi_dir 1 \
	--residual 1 \
	--ch_ind 1 \
	--embed_type 1 \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--learning_rate 0.00005

# DLinear on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model DLinear \
	--dataset ETTm1 \
	--individual \
	--learning_rate 0.00005

# iTransformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model iTransformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Transformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model Transformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Autoformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model Autoformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Flowformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model Flowformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Informer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model Informer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Reformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model Reformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# iFlashformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model iFlashformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# iFlowformer on ETTm1
./scripts/efficiency/run_efficiency_exp.sh \
	--model iFlowformer \
	--dataset ETTm1 \
	--learning_rate 0.00005
