#!/bin/bash
# Efficiency experiments for ECL dataset
# This script contains all model experiments for ECL dataset

# Experiments on ECL_96_336

# # MODE on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model MODE \
# 	--dataset ECL \
# 	--d_state 16 \
# 	--r_rank 8 \
# 	--d_conv 2 \
# 	--expand 2 \
# 	--ode_steps 10 \
# 	--learning_rate 0.0005

# # S_Mamba on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model S_Mamba \
# 	--dataset ECL \
# 	--d_state 32 \
# 	--learning_rate 0.0005

# BiMamba4TS on ECL
./scripts/efficiency/run_efficiency_exp.sh \
	--model BiMamba4TS \
	--dataset ECL \
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
	--learning_rate 0.0005

# PatchTST on ECL
./scripts/efficiency/run_efficiency_exp.sh \
	--model PatchTST \
	--dataset ECL \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.0005
	# --lradj 'TST' \

# DLinear on ECL
./scripts/efficiency/run_efficiency_exp.sh \
	--model DLinear \
	--dataset ECL \
	--individual \
	--learning_rate 0.0005

# # iTransformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model iTransformer \
# 	--dataset ECL \
# 	--learning_rate 0.0005

# # Transformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Transformer \
# 	--dataset ECL \
# 	--learning_rate 0.0005

# # Autoformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Autoformer \
# 	--dataset ECL \
# 	--learning_rate 0.0005

# # Flowformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Flowformer \
# 	--dataset ECL \
# 	--learning_rate 0.0005

# # Informer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Informer \
# 	--dataset ECL \
# 	--learning_rate 0.0005

# # Reformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
# 	--model Reformer \
# 	--dataset ECL \
# 	--learning_rate 0.0005

# # iFlashformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
#	--model iFlashformer \
#	--dataset ECL \
#	--learning_rate 0.0005

# # iFlowformer on ECL
# ./scripts/efficiency/run_efficiency_exp.sh \
#	--model iFlowformer \
#	--dataset ECL \
#	--learning_rate 0.0005
