"""
    Roodmus:
    A tool to generate cryo-EM images from MD trajectories using Parakeet

"""

import argparse

import roodmus.simulation.run_parakeet
import roodmus.trajectory.conformations_sampling
import roodmus.analysis.plot_ctf
import roodmus.analysis.plot_picking

# import analysis.analyse_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.1",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        help="additional help",
    )
    subparsers.required = True

    modules = [
        roodmus.simulation.run_parakeet,
        roodmus.trajectory.conformations_sampling,
        roodmus.analysis.plot_ctf,
        roodmus.analysis.plot_picking,
    ]

    for module in modules:
        this_parser = subparsers.add_parser(
            module.get_name(),
            help=module.__doc__,
        )
        module.add_arguments(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
