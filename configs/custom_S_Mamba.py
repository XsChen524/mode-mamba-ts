def add_s_mamba_parser(parser):
    """Add S_Mamba model-specific parameters to the parser."""
    # S_Mamba-specific parameters
    parser.add_argument('--d_state', type=int, default=16, help='SSM state expansion factor')
