"""
Command-line interface entry-point for all tasks related to estimation.

This is mostly a dispatch into the implemented algorithms.

"""

import sys
import argparse

from .quantile_iv import (
    configure_argparse as quantile_iv_configure_argparse,
    main as quantile_iv_main
)

from .deep_iv import (
    configure_argparse as deep_iv_configure_argparse,
    main as deep_iv_main
)

from .dfiv import (
    configure_argparse as dfiv_configure_argparse,
    main as dfiv_main
)

from .delivr import (
    configure_argparse as delivr_configure_argparse,
    main as delivr_main
)


def main():
    """Entry point for the estimation module.

    This is basically just a dispatch to specific algorithms.

    """
    parser = argparse.ArgumentParser(
        prog="ml-mr estimation"
    )

    algorithms = parser.add_subparsers(
        title="algorithm", dest="algorithm", required=True
    )

    quantile_iv_parser = algorithms.add_parser("quantile_iv")
    quantile_iv_configure_argparse(quantile_iv_parser)

    deep_iv_parser = algorithms.add_parser("deep_iv")
    deep_iv_configure_argparse(deep_iv_parser)

    dfiv_parser = algorithms.add_parser("dfiv")
    dfiv_configure_argparse(dfiv_parser)

    delivr_parser = algorithms.add_parser("delivr")
    delivr_configure_argparse(delivr_parser)

    args = parser.parse_args(sys.argv[2:])
    if args.algorithm == "quantile_iv":
        quantile_iv_main(args)
    elif args.algorithm == "deep_iv":
        deep_iv_main(args)
    elif args.algorithm == "dfiv":
        dfiv_main(args)
    elif args.algorithm == "delivr":
        delivr_main(args)
    else:
        raise ValueError("Invalid algorithm.")
