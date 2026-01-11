def add_crossformer_parser(parser):
    """Add Crossformer model-specific parameters to the parser."""
    # Crossformer requires embed_type to be set for positional embeddings
    parser.add_argument(
        "--embed_type",
        type=int,
        default=1,
        help="embedding type (required for Crossformer)",
    )

    # Patch embedding parameters (shared with PatchTST)
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--padding_patch", type=str, default="end", help="padding patch"
    )
