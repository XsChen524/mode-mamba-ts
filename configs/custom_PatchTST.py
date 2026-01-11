def add_patchtst_parser(parser):
    """Add PatchTST model-specific parameters to the parser."""
    # PatchTST-specific parameters
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.3,
        help="percentage of training for learning rate warmup (for OneCycleLR)",
    )
    parser.add_argument(
        "--patch_len", type=int, default=16, help="Length of each patch"
    )
    parser.add_argument("--stride", type=int, default=8, help="Stride between patches")
    parser.add_argument(
        "--padding_patch", type=str, default="end", help="Padding mode for patches"
    )

    # dropout parameters - fc_dropout and head_dropout specific to PatchTST
    parser.add_argument(
        "--fc_dropout",
        type=float,
        default=0.2,
        help="Dropout rate in fully connected layers of head",
    )
    parser.add_argument(
        "--head_dropout", type=float, default=0.0, help="Dropout rate in head"
    )

    # decomposition parameters (for decomposition variant)
    parser.add_argument(
        "--decomposition",
        action="store_true",
        help="Use series decomposition",
        default=False,
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size of moving average for decomposition",
    )

    # attention parameters
    parser.add_argument(
        "--res_attention",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use residual attention",
    )
    parser.add_argument(
        "--pre_norm",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use pre-normalization",
    )
    parser.add_argument(
        "--pos_embed_type", type=str, default="sincos", help="Positional embedding type"
    )
