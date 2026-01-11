def add_mode_parser(parser):
    """Add MODE model-specific parameters to the parser."""

    # MODE-specific parameters
    parser.add_argument(
        "--d_state", type=int, default=16, help="parameter of MODE state dimension"
    )
    parser.add_argument("--r_rank", type=int, default=8, help="parameter of MODE rank")
    parser.add_argument(
        "--d_conv", type=int, default=2, help="parameter of MODE conv kernel size"
    )
    parser.add_argument(
        "--expand", type=int, default=2, help="parameter of MODE expansion factor"
    )
    parser.add_argument(
        "--ode_solver",
        type=str,
        default="euler",
        help="ODE solver method for MODE, options: [euler, dopri5, rk4, adams, explicit_adams]",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=10,
        help="number of ODE integration steps for MODE",
    )
    parser.add_argument(
        "--hippo", action="store_true", help="whether to use hippo init"
    )
    parser.add_argument(
        "--ode_type",
        type=str,
        default="static",
        help="""type of ode, options: [
            "none",
            "static",
            "dynamic"
        ]""",
    )
    parser.add_argument(
        "--replace_block",
        type=str,
        default="none",
        help="""replace ode-enhanced block, options: [
            "none",
            "s-mamba",
            "attention-ffn",
            "linear"
        ]""",
    )
