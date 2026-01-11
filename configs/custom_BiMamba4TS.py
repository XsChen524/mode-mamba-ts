def add_bimamba4ts_parser(parser):
    """Add BiMamba4TS model-specific parameters to the parser."""
    # token embedding
    parser.add_argument('--d_model',        type=int,   default=64,  help='Sequence Elements embedding dimension')
    parser.add_argument('--d_ff',           type=int,   default=128, help='Second Embedded representation')

    # mamba block
    parser.add_argument('--bi_dir',         type=int,   default=1,   help='use bidirectional Mamba?')
    parser.add_argument('--d_state',        type=int,   default=32,  help='d_state parameter of Mamba')
    parser.add_argument('--d_conv',         type=int,   default=2,   help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact',         type=int,   default=1,   help='expand factor parameter of Mamba')

    parser.add_argument('--e_layers',       type=int,   default=1,   help='layers of encoder')
    parser.add_argument('--dropout',        type=float, default=0.2, help='dropout')
    parser.add_argument('--activation',     type=str,   default='gelu', help='activation')

    # channel independence (specific to BiMamba4TS)
    parser.add_argument("--SRA",            action="store_true", default=False, help="use series-relation-aware decider?")
    parser.add_argument("--threshold",      type=float, default=0.6, help="threshold for SRA")
    parser.add_argument("--ch_ind",         type=int,   default=1,   help="forced channel independent?")
    parser.add_argument("--residual",       type=int,   default=1,   help="residual connection?")

    # patching parameters
    parser.add_argument("--patch_len",      type=int,   default=16,  help="patch length")
    parser.add_argument("--stride",         type=int,   default=8,   help="stride")
    parser.add_argument("--padding_patch",  type=str,   default="end", help="padding patch")

    # Model settings
    parser.add_argument("--embed_type",     type=int,   default=1,   help="embedding type")
    parser.add_argument("--pos_embed_type", type=str,   default="sincos", help="positional embedding type")
    parser.add_argument("--pos_learnable",  type=int,   default=0,   help="learnable positional embedding")
