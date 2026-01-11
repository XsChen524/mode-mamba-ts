#!/bin/bash
# Efficiency experiments for Weather dataset
# This script contains all model experiments for Weather dataset

# MODE on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model MODE \
	--dataset Weather \
	--d_state 4 \
	--r_rank 1 \
	--d_conv 2 \
	--expand 2 \
	--ode_steps 2 \
	--learning_rate 0.00006

# PatchTST on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model PatchTST \
	--dataset Weather \
	--patch_len 16 \
	--stride 8 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.0001
	# --lradj 'TST' \

# S_Mamba on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model S_Mamba \
	--dataset Weather \
	--d_state 2 \
	--learning_rate 0.00005

# BiMamba4TS on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model BiMamba4TS \
	--dataset Weather \
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

# DLinear on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model DLinear \
	--dataset Weather \
	--individual \
	--learning_rate 0.00005

# iTransformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model iTransformer \
	--dataset Weather \
	--learning_rate 0.00005

# Transformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model Transformer \
	--dataset Weather \
	--learning_rate 0.00005

# Autoformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model Autoformer \
	--dataset Weather \
	--learning_rate 0.00005

# Flowformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model Flowformer \
	--dataset Weather \
	--learning_rate 0.00005

# Informer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model Informer \
	--dataset Weather \
	--learning_rate 0.00005

# Reformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model Reformer \
	--dataset Weather \
	--learning_rate 0.00005

# iFlashformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model iFlashformer \
	--dataset Weather \
	--learning_rate 0.00005

# iFlowformer on Weather
./scripts/efficiency/run_efficiency_exp.sh \
	--model iFlowformer \
	--dataset Weather \
	--learning_rate 0.00005
