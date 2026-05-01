import argparse

from atommovr.utils.benchmarking import Benchmarking, BenchmarkingFigure
from atommovr.utils.core import Configurations, PhysicalParams
from atommovr.algorithms.single_species import (
    PCFA,
    Hungarian,
    BalanceAndCompact,
    BCv2,
    ParallelLBAP,
    ParallelHungarian,
    GeneralizedBalance,
    Tetris,
    BlindCompress,
    BlindShell,
    BlindSweep,
)
from atommovr.algorithms.dual_species import InsideOut, NaiveParHung
from atommovr.utils.errormodels import (
    ZeroNoise,
    UniformVacuumTweezerError,
    YbRydbergAODErrorModel,
)


def main():
    parser = argparse.ArgumentParser(description="Run algorithm benchmarks")
    parser.add_argument(
        "--min_size", type=int, default=10, help="Minimum L for square target"
    )
    parser.add_argument(
        "--max_size", type=int, default=14, help="Maximum L for square target"
    )
    parser.add_argument(
        "--step_size", type=int, default=1, help="Step between L values"
    )
    parser.add_argument(
        "--shots", type=int, default=50, help="Number of shots per configuration"
    )
    parser.add_argument("--rounds", type=int, default=1, help="Rearrangement rounds")
    parser.add_argument(
        "--save", action="store_true", help="Save xarray results to data/"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Save name (without extension)"
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Benchmark dual-species algorithms (InsideOut, NaiveParHung) "
        "with n_species=2 and CHECKERBOARD target",
    )
    parser.add_argument(
        "--ybryd-only",
        action="store_true",
        help="Use only YbRydbergAODErrorModel (skip ZeroNoise and Uniform)",
    )
    args = parser.parse_args()

    if args.dual:
        algos = [InsideOut(), NaiveParHung()]
        targets = [Configurations.CHECKERBOARD]
        n_species = 2
    else:
        algos = [
            PCFA(),
            BCv2(),
            Hungarian(),
            BalanceAndCompact(),
            ParallelLBAP(),
            ParallelHungarian(),
            GeneralizedBalance(),
            Tetris(),
            BlindCompress(),
            BlindShell(),
            BlindSweep(),
        ]
        targets = [Configurations.MIDDLE_FILL]
        n_species = 1

    params = [PhysicalParams()]
    sizes = list(range(args.min_size, args.max_size + 1, args.step_size))

    if args.name is None:
        args.name = f"benchmark_{'_'.join(algo.__class__.__name__ for algo in algos)}"

    if getattr(args, "ybryd_only", False):
        error_models = [YbRydbergAODErrorModel()]
    else:
        error_models = [
            ZeroNoise(),
            UniformVacuumTweezerError(),
            YbRydbergAODErrorModel(),
        ]

    fig = BenchmarkingFigure(
        variables=["Time", "Mean moves", "Parallel move batches", "Success rate"],
        figure_type="scale",
    )
    bench = Benchmarking(
        algos=algos,
        target_configs=targets,
        error_models_list=error_models,
        phys_params_list=params,
        sys_sizes=sizes,
        rounds_list=[args.rounds],
        n_shots=args.shots,
        n_species=n_species,
        figure_output=fig,
        per_round_logging=True,
        check_sufficient_atoms=True,
        show_progress=True,
    )

    bench.run(do_ejection=False)
    if args.save:
        bench.save(args.name)

    # Optionally, plot results: comment in if needed
    bench.plot_results(save=True, savename=f"{args.name}_plot")


if __name__ == "__main__":
    main()
