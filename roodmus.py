""" Roodmus: a tool to generate cryo-EM images from MD trajectories using Parakeet"""

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")    
    
    subparsers = parser.add_subparsers(title="subcommands", description="valid subcommands", help="additional help")
    subparsers.required = True
    
    import roodmus.run_parakeet.run_parakeet
    import roodmus.trajectory.waymarking
    import roodmus.analysis.compare_ctf
    
    modules = [
        roodmus.run_parakeet.run_parakeet,
        roodmus.trajectory.waymarking,
        roodmus.analysis.compare_ctf,
    ]

    def get_string_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]
    
    for module in modules:
        print(get_string_name(module))
        print(module.get_name())
        print(module.__file__)
        this_parser = subparsers.add_parser(module.get_name(), help=module.__doc__)
        module.add_arguments(this_parser)
        this_parser.set_defaults(func=module.main)
        
    args = parser.parse_args()
    args.func(args)
    
if __name__ == "__main__":
    main()
    

