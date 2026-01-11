import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from configs.model_parser_dict import model_parser_dict


if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="iTransformer")

    # Always load common arguments first
    if "__all__" in model_parser_dict:
        for common_parser in model_parser_dict["__all__"]:
            common_parser(parser=parser)

    # Add model-specific arguments if model is specified
    temp_args, _ = parser.parse_known_args()
    if temp_args.model and temp_args.model in model_parser_dict:
        # Load model-specific parsers
        for model_parser in model_parser_dict[temp_args.model]:
            model_parser(parser=parser)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.exp_name == "partial_train":
        Exp = Exp_Long_Term_Forecast_Partial
    else:
        Exp = Exp_Long_Term_Forecast

    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.is_training == 1:
        for ii in range(args.itr):
            # setting record of experiments
            robustness_str = ""
            if args.noise_level > 0:
                robustness_str += "_nl:{}".format(args.noise_level)

            if args.model == "MODE":
                setting = "{}_{}_des:{}_dst:{}_rr:{}_dc:{}_exp:{}_odes:{}_hip:{}_odt:{}_rb:{}_lr:{}_ft:{}_sl:{}_ll:{}_pl:{}_dm:{}_nh:{}_el:{}_dl:{}_df:{}_fc:{}_eb:{}_dt:{}_cs:{}{}_itr:{}".format(
                    args.model_id,
                    args.model,
                    args.des,
                    args.d_state,
                    args.r_rank,
                    args.d_conv,
                    args.expand,
                    args.ode_steps,
                    args.hippo,
                    args.ode_type,
                    args.replace_block,
                    args.learning_rate,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.class_strategy,
                    robustness_str,
                    ii,
                )
            else:
                # For non-Mamba models, exclude Mamba-specific parameters
                setting = "{}_{}_des:{}_lr:{}_ft:{}_sl:{}_ll:{}_pl:{}_dm:{}_nh:{}_el:{}_dl:{}_df:{}_fc:{}_eb:{}_dt:{}_cs:{}{}_itr:{}".format(
                    args.model_id,
                    args.model,
                    args.des,
                    args.learning_rate,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.class_strategy,
                    robustness_str,
                    ii,
                )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)

            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        for ii in range(args.itr):
            # setting record of experiments
            # Include robustness parameters if they are non-default
            robustness_str = ""
            if args.noise_level > 0:
                robustness_str += "_nl:{}".format(args.noise_level)

            if args.model == "MODE":
                setting = "{}_{}_des:{}_dst:{}_rr:{}_dc:{}_exp:{}_odes:{}_hip:{}_odt:{}_rb:{}_lr:{}_ft:{}_sl:{}_ll:{}_pl:{}_dm:{}_nh:{}_el:{}_dl:{}_df:{}_fc:{}_eb:{}_dt:{}_cs:{}{}_itr:{}".format(
                    args.model_id,
                    args.model,
                    args.des,
                    args.d_state,
                    args.r_rank,
                    args.d_conv,
                    args.expand,
                    args.ode_steps,
                    args.hippo,
                    args.ode_type,
                    args.replace_block,
                    args.learning_rate,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.class_strategy,
                    robustness_str,
                    ii,
                )
            else:
                # For non-Mamba models, exclude Mamba-specific parameters
                setting = "{}_{}_des:{}_lr:{}_ft:{}_sl:{}_ll:{}_pl:{}_dm:{}_nh:{}_el:{}_dl:{}_df:{}_fc:{}_eb:{}_dt:{}_cs:{}{}_itr:{}".format(
                    args.model_id,
                    args.model,
                    args.des,
                    args.learning_rate,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.class_strategy,
                    robustness_str,
                    ii,
                )
            exp = Exp(args)
        ii = 0
        if args.model == "MODE":
            setting = "{}_{}_dst:{}_rr:{}_dc:{}_exp:{}_odes:{}_hip:{}_odt:{}_rb:{}_lr:{}_ft:{}_sl:{}_ll:{}_pl:{}_dm:{}_nh:{}_el:{}_dl:{}_df:{}_fc:{}_eb:{}_dt:{}_des:{}_cs:{}_itr:{}".format(
                args.model_id,
                args.model,
                args.des,
                args.d_state,
                args.r_rank,
                args.d_conv,
                args.expand,
                args.ode_steps,
                args.hippo,
                args.ode_type,
                args.replace_block,
                args.learning_rate,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.class_strategy,
                ii,
            )
        else:
            setting = "{}_{}_des:{}_lr:{}_ft:{}_sl:{}_ll:{}_pl:{}_dm:{}_nh:{}_el:{}_dl:{}_df:{}_fc:{}_eb:{}_dt:{}_cs:{}_itr:{}".format(
                args.model_id,
                args.model,
                args.des,
                args.learning_rate,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.class_strategy,
                ii,
            )

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
