"""
Model parameter parser dictionary.
Maps model names to their specific parameter parser functions.
"""

from configs.common import add_common_args
from configs.custom_iTransformer import add_iTransformer_parser
from configs.custom_S_Mamba import add_s_mamba_parser
from configs.custom_MODE import add_mode_parser
from configs.custom_PatchTST import add_patchtst_parser
from configs.custom_DLinear import add_dlinear_parser
from configs.custom_BiMamba4TS import add_bimamba4ts_parser
from configs.custom_Crossformer import add_crossformer_parser

model_parser_dict = {
    # All models start with common arguments
    "__all__": [add_common_args],
    # iTransformer family - iTransformer-specific parameters + common
    "iTransformer": [add_iTransformer_parser],
    "iInformer": [add_iTransformer_parser],
    "iReformer": [add_iTransformer_parser],
    "iFlowformer": [add_iTransformer_parser],
    "iFlashformer": [add_iTransformer_parser],
    # Transformer variants - only common parameters
    "Transformer": [],
    "Transformer_M": [],
    "Informer": [],
    "Informer_M": [],
    "Reformer": [],
    "Reformer_M": [],
    "Autoformer": [],
    "Autoformer_M": [],
    "Flowformer": [],
    "Flashformer_M": [],
    "Flashformer": [],
    "Flowformer_M": [],
    # Original Mamba-based models - Mamba common + model-specific
    "S_Mamba": [add_s_mamba_parser],
    "MODE": [add_mode_parser],
    # New benchmark models
    "PatchTST": [add_patchtst_parser],
    "DLinear": [add_dlinear_parser],
    "Crossformer": [add_crossformer_parser],
    "BiMamba4TS": [add_bimamba4ts_parser],
}

# Aliases for common models (for backward compatibility)
model_aliases = {
    "MODE": "MODE",
    "S_Mamba": "S_Mamba",
    "iTransformer": "iTransformer",
    "iInformer": "iInformer",
    "iReformer": "iReformer",
    "iFlowformer": "iFlowformer",
    "iFlashformer": "iFlashformer",
    "Transformer": "Transformer",
    "Transformer_M": "Transformer_M",
    "Informer": "Informer",
    "Informer_M": "Informer_M",
    "Reformer": "Reformer",
    "Reformer_M": "Reformer_M",
    "Flowformer": "Flowformer",
    "Flashformer_M": "Flashformer_M",
    "Flashformer": "Flashformer",
    "Flowformer_M": "Flowformer_M",
    "Autoformer": "Autoformer",
    "Autoformer_M": "Autoformer_M",
    "PatchTST": "PatchTST",
    "DLinear": "DLinear",
    "Crossformer": "Crossformer",
    "BiMamba4TS": "BiMamba4TS",
}
