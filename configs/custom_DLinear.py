def add_dlinear_parser(parser):
    """Add DLinear model-specific parameters to the parser."""
    parser.add_argument('--individual', action='store_true', default=False, help='individual linear layers for each channel')
