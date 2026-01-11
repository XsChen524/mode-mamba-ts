#!/bin/bash
# Efficiency experiments for PEMS08 dataset
# This script contains all model experiments for PEMS08 dataset

# # MODE on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model MODE \
# 	--dataset PEMS08 \
# 	--d_state 32 \
# 	--r_rank 16 \
# 	--d_conv 2 \
# 	--expand 2 \
# 	--ode_steps 10 \
# 	--learning_rate 0.0015

# # S_Mamba on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model S_Mamba \
# 	--dataset PEMS08 \
# 	--d_state 32 \
# 	--learning_rate 0.001

# BiMamba4TS on PEMS08
./scripts/efficiency/run_efficiency_exp.sh \
	--model BiMamba4TS \
	--dataset PEMS08 \
	--d_state 32 \
	--d_conv 2 \
	--e_fact 2 \
	--bi_dir 1 \
	--residual 1 \
	--ch_ind 1 \
	--embed_type 1 \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--learning_rate 0.001

# PatchTST on PEMS08
./scripts/efficiency/run_efficiency_exp.sh \
	--model PatchTST \
	--dataset PEMS08 \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.001
	# --lradj 'TST' \

# DLinear on PEMS08
./scripts/efficiency/run_efficiency_exp.sh \
	--model DLinear \
	--dataset PEMS08 \
	--individual \
	--learning_rate 0.001

# # iTransformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model iTransformer \
# 	--dataset PEMS08 \
# 	--learning_rate 0.001

# # Transformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Transformer \
# 	--dataset PEMS08 \
# 	--learning_rate 0.001

# # Autoformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Autoformer \
# 	--dataset PEMS08 \
# 	--learning_rate 0.001

# # Flowformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Flowformer \
# 	--dataset PEMS08 \
# 	--learning_rate 0.001

# # Informer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Informer \
# 	--dataset PEMS08 \
# 	--learning_rate 0.001

# # Reformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Reformer \
# 	--dataset PEMS08 \
# 	--learning_rate 0.001

# # iFlashformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
#	--model iFlashformer \
#	--dataset PEMS08 \
#	--learning_rate 0.001

# # iFlowformer on PEMS08
# ./scripts/efficiency/run_efficiency_exp.sh \
#	--model iFlowformer \
#	--dataset PEMS08 \
#	--learning_rate 0.001
