# Object for running benchmarking rounds and saving data

import math
import csv
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from atommovr.utils.errormodels import ZeroNoise
from atommovr.utils.core import (
    generate_random_target_configs,
    generate_random_init_configs,
    PhysicalParams,
    Configurations,
    CONFIGURATION_PLOT_LABELS,
    array_shape_for_geometry,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.algorithms.Algorithm_class import Algorithm, get_effective_target_grid
from atommovr.algorithms.source.scaling_lower_bound import calculate_Zstar_better

try:
    from tqdm.auto import tqdm as tqdm_auto
except ImportError:  # pragma: no cover - tqdm optional
    tqdm_auto = None


def evaluate_moves(array: AtomArray, move_list: list):
    # making reference time
    t_total = 0
    N_parallel_moves = 0
    N_non_parallel_moves = 0

    # iterating through moves and updating matrix
    for _, move_set in enumerate(move_list):

        # performing the move
        [_, _], move_time = array.move_atoms(move_set)
        N_parallel_moves += 1
        N_non_parallel_moves += len(move_set)

        # calculating the time to complete the move set in parallel
        t_total += move_time

    return array, float(t_total), [N_parallel_moves, N_non_parallel_moves]


class BenchmarkingFigure:
    """

    NB: this is a placeholder class to mark an opportunity for future feature development (see CONTRIBUTING.md). It is not currently operational.

    Class that specifies plot parameters and figure types to be used in conjunction with the `Benchmarking` class.

    This class just specifies what you want to plot, to actually plot you have to pass it to an instance of the
    `Benchmarking` class and call the `plot_results()` function.

    ## Parameters
    - `y_axis_variables` (list):
        the observables to plot. Must be in ['Success rate', 'Filling fraction', 'Time', 'Wrong places #', 'Total atoms']
    - `figure_type` (str):
        The kind of figure you want to make. Options are histogram ('hist'), a plot comparing different algorithms ('scale'), or a plot comparing different target configurations for the same algorithm ('pattern').
    """

    def __init__(self, variables: list = ["Success rate"], figure_type: str = "scale"):
        # Maintain user-facing canonical labels but map them to dataset keys
        lower_to_canonical = {
            "success rate": "Success rate",
            "filling fraction": "Filling fraction",
            "time": "Time",
            "wall time": "Wall time",
            "walltime": "Wall time",
            "wrong places #": "Wrong places #",
            "wrong places": "Wrong places #",
            "total atoms": "Total atoms",
            "n atoms": "Total atoms",
            "mean moves": "Mean moves",
            "moves": "Mean moves",
            "parallel move batches": "Parallel move batches",
            "parallel batches": "Parallel move batches",
        }
        canonical_to_dskey = {
            "Success rate": "success rate",
            "Filling fraction": "filling fraction",
            "Time": "time",
            "Wall time": "wall time",
            "Wrong places #": "wrong places",
            "Total atoms": "n atoms",
            "Mean moves": "mean moves",
            "Parallel move batches": "parallel move batches",
        }

        normalized_user: list[str] = []
        self._yuser_to_dskey: dict[str, str] = {}
        for variable in variables:
            v = str(variable).strip()
            if v == "":
                continue
            lower = v.lower()
            canonical = lower_to_canonical.get(lower)
            if canonical is None:
                allowed = list(canonical_to_dskey.keys())
                raise KeyError(
                    f"Variable '{variable}' is not recognized. Allowed: {allowed}."
                )
            normalized_user.append(canonical)
            self._yuser_to_dskey[canonical] = canonical_to_dskey[canonical]

        # Store user-facing labels for plotting and lookups; keep original capitalization
        self.y_axis_variables = normalized_user
        self.figure_type = figure_type

    def generate_scaling_figure(
        self,
        x_axis_unused,
        benchmarking_results,
        title,
        x_label,
        save,
        savename="Algorithm_scaling",
        complexity_summary=None,
        analytical_model_fns=None,
    ):
        sns.set_theme(style="whitegrid", font_scale=1.15)

        def _format_axis(ax_obj, metric_key):
            label_map = {
                "time": "Mean success time (s)",
                "wall time": "Wall time (s)",
                "success rate": "Success rate",
                "filling fraction": "Filling fraction",
                "wrong places": "Wrong places",
                "n atoms": "Total atoms",
                "mean moves": "Mean moves per shot",
                "parallel move batches": "Parallel move batches",
            }
            ylabel = label_map.get(metric_key, metric_key.capitalize())
            ax_obj.set_xlabel("Target sites (atoms)")
            ax_obj.set_ylabel(ylabel)
            if title is not None:
                ax_obj.set_title(title)
            ax_obj.legend(frameon=False, loc="best")
            ax_obj.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
            ax_obj.tick_params(axis="both", labelsize=11)
            sns.despine(ax=ax_obj)

        def _prepare_xy(xs, ys):
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(mask):
                return None
            xs = xs[mask]
            ys = ys[mask]
            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]
            return xs, ys

        def _power_law_fit(xs, ys):
            prepared = _prepare_xy(xs, ys)
            if prepared is None:
                return None
            xs, ys = prepared
            mask = (xs > 0) & (ys > 0)
            if np.count_nonzero(mask) < 2:
                return None
            log_x = np.log(xs[mask])
            log_y = np.log(ys[mask])
            slope, intercept = np.polyfit(log_x, log_y, 1)
            coef = float(np.exp(intercept))
            exponent = float(slope)
            return coef, exponent

        def _plot_series(
            ax_obj,
            xs,
            ys,
            label,
            color,
            fit_fn=None,
            force_power_fit=False,
            connect_dots=False,
        ):
            prepared = _prepare_xy(xs, ys)
            if prepared is None:
                return False
            xs, ys = prepared
            # Use matplotlib Axes.scatter to avoid triggering Seaborn/pandas internals when tests
            # provide MagicMock axes via patched `plt.subplots`.
            try:
                ax_obj.scatter(
                    xs,
                    ys,
                    color=color,
                    s=60,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.4,
                    label=label,
                )
            except Exception:
                # Fallback to seaborn if axes implement what seaborn expects
                sns.scatterplot(
                    x=xs,
                    y=ys,
                    ax=ax_obj,
                    color=color,
                    s=60,
                    marker="o",
                    edgecolor="black",
                    linewidth=0.4,
                    label=label,
                )

            if connect_dots:
                # Sort by x to connect in order
                sort_idx = np.argsort(xs)
                ax_obj.plot(
                    xs[sort_idx],
                    ys[sort_idx],
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.6,
                )
                return True

            if fit_fn is not None and xs.size >= 2:
                x_fit = np.linspace(xs.min(), xs.max(), 200)
                x_fit = x_fit[np.isfinite(x_fit) & (x_fit > 0)]
                if x_fit.size >= 2:
                    ax_obj.plot(
                        x_fit,
                        fit_fn(x_fit),
                        color=color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.85,
                    )
            elif force_power_fit and xs.size >= 2:
                params = _power_law_fit(xs, ys)
                if params is not None:
                    coef, exponent = params
                    x_fit = np.linspace(xs.min(), xs.max(), 200)
                    x_fit = x_fit[np.isfinite(x_fit) & (x_fit > 0)]
                    if x_fit.size >= 2:
                        ax_obj.plot(
                            x_fit,
                            coef * np.power(x_fit, exponent),
                            color=color,
                            linestyle="-",
                            linewidth=1.8,
                            alpha=0.85,
                        )
            if fit_fn is None and (not force_power_fit) and xs.size >= 2:
                coeffs = np.polyfit(xs, ys, 1)
                ax_obj.plot(
                    xs,
                    np.polyval(coeffs, xs),
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.75,
                )
            return True

        def _slugify(name: str) -> str:
            cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
            cleaned = cleaned.strip("_")
            return cleaned or "metric"

        def _normalize_target_label(target_entry, index: int) -> str:
            if hasattr(target_entry, "name"):
                return str(target_entry.name)
            if isinstance(target_entry, np.ndarray):
                return f"Custom{index}"
            return str(target_entry)

        figs_dir = Path("./figs")
        figs_dir.mkdir(parents=True, exist_ok=True)

        has_complexity_data = bool(complexity_summary)

        if isinstance(benchmarking_results, xr.Dataset):
            ds = benchmarking_results
            algo_labels = list(ds.coords["algorithm"].values)
            palette = sns.color_palette("colorblind", max(len(algo_labels), 1))
            color_map = {
                algo: palette[idx % len(palette)]
                for idx, algo in enumerate(algo_labels)
            }

            target_values = list(ds.coords["target"].values)
            target_sel = target_values[0]
            err_values = list(ds.coords["error model"].values)
            phys_sel = ds.coords["physical params"].values[0]
            round_sel = ds.coords["num rounds"].values[0]

            target_label_map = {
                idx: _normalize_target_label(val, idx)
                for idx, val in enumerate(target_values)
            }
            target_label = target_label_map.get(0, str(target_sel))
            phys_label = str(phys_sel)
            rounds_value = int(round(round_sel))
            do_ejection_attr = benchmarking_results.attrs.get("do_ejection", None)

            complexity_records = complexity_summary or []
            metric_name_map = {
                "time": "wall_time",
                "wall time": "wall_time",
                "mean moves": "moves",
                "parallel move batches": "parallel_batches",
            }

            def _select_complexity_fit(
                metric_key: str, algo_name: str, error_label: str
            ):
                if not complexity_records or metric_key not in metric_name_map:
                    return None
                desired_metric = metric_name_map[metric_key]
                for rec in complexity_records:
                    if rec.get("metric_name") != desired_metric:
                        continue
                    if rec.get("scale_axis") != "target_sites":
                        continue
                    if rec.get("algorithm") != algo_name:
                        continue
                    if rec.get("target") != target_label:
                        continue
                    if rec.get("error_model") != error_label:
                        continue
                    if rec.get("physical_params_label") != phys_label:
                        continue
                    if int(rec.get("num_rounds", rounds_value)) != rounds_value:
                        continue
                    if do_ejection_attr is not None and rec.get("do_ejection") != bool(
                        do_ejection_attr
                    ):
                        continue
                    coef = rec.get("coefficient")
                    exponent = rec.get("exponent")
                    if coef is None or exponent is None:
                        continue
                    if coef <= 0:
                        continue
                    return rec
                return None

            for err_sel in err_values:
                error_label = str(err_sel)
                base_sel = {
                    "target": target_sel,
                    "error model": err_sel,
                    "physical params": phys_sel,
                    "num rounds": round_sel,
                }
                target_counts_da = ds["n targets"].sel(**base_sel)

                for y_var in self.y_axis_variables:
                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    plotted_any = False
                    ds_key = self._yuser_to_dskey.get(y_var, y_var.lower())
                    da = ds[ds_key].sel(**base_sel)
                    is_success_rate = ds_key == "success rate"

                    for algo in algo_labels:
                        y_vals = da.sel(algorithm=algo).values.reshape(-1)
                        x_vals = target_counts_da.sel(algorithm=algo).values.reshape(-1)
                        fit_rec = _select_complexity_fit(ds_key, str(algo), error_label)
                        fit_fn = None
                        if fit_rec is not None and not is_success_rate:
                            coef = float(fit_rec.get("coefficient"))
                            exponent = float(fit_rec.get("exponent"))
                            fit_fn = lambda arr, c=coef, e=exponent: c * np.power(
                                arr, e
                            )

                        plotted_any |= _plot_series(
                            ax,
                            x_vals,
                            y_vals,
                            str(algo),
                            color_map[algo],
                            fit_fn=fit_fn,
                            force_power_fit=(
                                has_complexity_data and not is_success_rate
                            ),
                            connect_dots=is_success_rate,
                        )

                    if is_success_rate and analytical_model_fns:
                        for algo in algo_labels:
                            key = (str(algo), error_label)
                            model_fn = analytical_model_fns.get(key)
                            if model_fn is None:
                                continue
                            x_vals = target_counts_da.sel(
                                algorithm=algo
                            ).values.reshape(-1)
                            x_valid = x_vals[np.isfinite(x_vals) & (x_vals > 0)]
                            if x_valid.size < 2:
                                continue
                            x_smooth = np.linspace(x_valid.min(), x_valid.max(), 200)
                            y_smooth = model_fn(x_smooth)
                            ax.plot(
                                x_smooth,
                                y_smooth,
                                color=color_map[algo],
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.7,
                            )

                    if not plotted_any:
                        plt.close(fig)
                        continue
                    _format_axis(ax, ds_key)
                    fig.tight_layout()

                    if save:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        plt.savefig(
                            figs_dir
                            / f"{savename}_{_slugify(error_label)}_{_slugify(y_var)}_{timestamp}.svg",
                            format="svg",
                            dpi=300,
                        )
                    plt.close(fig)
            return

        # Legacy list-of-dicts fallback
        palette = sns.color_palette("colorblind", 10)
        color_map = {}
        color_index = 0

        block_size = len(x_axis_unused) if x_axis_unused is not None else 0
        for y_var in self.y_axis_variables:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            plotted_any = False
            n_datapoints_added = 0
            y_axis_vals = []
            target_vals = []

            for algo_results in benchmarking_results:
                value = algo_results[y_var]
                # If an explicit x-axis was provided, use it in round-robin for legacy list-of-dicts
                if block_size > 0:
                    idx = n_datapoints_added % block_size
                    target_value = x_axis_unused[idx]
                else:
                    target_value = algo_results.get("n targets")

                if isinstance(value, list):
                    value = float(np.mean(value))
                if isinstance(target_value, list):
                    target_value = float(np.mean(target_value))
                if target_value is None:
                    raise KeyError(
                        "Legacy benchmarking results must include 'n targets' to plot against target sites."
                    )
                if math.isnan(value) or math.isnan(target_value):
                    raise Exception(
                        "Data to plot contains nan (NaN) in values to plot; aborting."
                    )
                y_axis_vals.append(value)
                target_vals.append(target_value)
                n_datapoints_added += 1

                if block_size > 0 and n_datapoints_added % block_size == 0:
                    algo_obj = algo_results["algorithm"]
                    algo_label = getattr(
                        getattr(algo_obj, "__class__", type(algo_obj)),
                        "__name__",
                        str(algo_obj),
                    )
                    if algo_label not in color_map:
                        color_map[algo_label] = palette[color_index % len(palette)]
                        color_index += 1
                    plotted_any |= _plot_series(
                        ax,
                        target_vals,
                        y_axis_vals,
                        algo_label,
                        color_map[algo_label],
                        force_power_fit=has_complexity_data,
                    )
                    y_axis_vals = []
                    target_vals = []

            if block_size == 0 and y_axis_vals and target_vals:
                algo_obj = benchmarking_results[-1]["algorithm"]
                algo_label = getattr(
                    getattr(algo_obj, "__class__", type(algo_obj)),
                    "__name__",
                    str(algo_obj),
                )
                if algo_label not in color_map:
                    color_map[algo_label] = palette[color_index % len(palette)]
                    color_index += 1
                plotted_any |= _plot_series(
                    ax,
                    target_vals,
                    y_axis_vals,
                    algo_label,
                    color_map[algo_label],
                    force_power_fit=has_complexity_data,
                )

            if not plotted_any:
                plt.close(fig)
                continue
            _format_axis(ax, self._yuser_to_dskey.get(y_var, y_var.lower()))
            fig.tight_layout()
            if save:
                plt.savefig(
                    figs_dir / f"{savename}_{_slugify(y_var)}.svg",
                    format="svg",
                    dpi=300,
                )
            plt.close(fig)

    def generate_histogram_figure(
        self, benchmarking_results, title, x_label, save=False, savename="Histogram"
    ):
        # Prepare subplots for each requested variable
        n_vars = len(self.y_axis_variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(5, 5 * max(1, n_vars)))
        if n_vars == 1:
            axes = [axes]

        for varind, y_var in enumerate(self.y_axis_variables):
            hist_data = []
            algos_name = []
            for algo_results in benchmarking_results:
                hist_data.append(algo_results[y_var])
                algos_name.append(str(algo_results["algorithm"]))

            ax = axes[varind]
            ax.set_xlabel(y_var)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{y_var} histogram")
            try:
                ax.hist(hist_data, bins=10, label=algos_name)
                ax.legend()
            except TypeError:
                # Fallback to single-axis plotting
                fig_single, ax_single = plt.subplots(figsize=(5, 5))
                ax_single.set_xlabel(y_var)
                ax_single.set_ylabel("Frequency")
                ax_single.set_title(f"{y_var} histogram")
                ax_single.hist(hist_data, bins=10, label=algos_name)
                ax_single.legend()

        if save:
            plt.savefig(f"./figs/{savename}")

    def generate_pattern_figure(
        self,
        x_axis,
        benchmarking_results,
        title,
        x_label,
        save=False,
        savename="Pattern_scaling",
    ):

        fig, ax = plt.subplots(
            len(self.y_axis_variables), 1, figsize=(5, 5 * len(self.y_axis_variables))
        )
        # Iterate over the y-axis variables
        for varind, y_var in enumerate(self.y_axis_variables):
            separate_pattern_flag = 0
            y_axis = []

            # Iterate over the benchmarking results of each target pattern
            for pattern_results in benchmarking_results:

                # If the y-axis variable is a list (e.g. filling fraction), take its average
                if type(pattern_results[y_var]) is list:
                    pattern_results[y_var] = np.mean(pattern_results[y_var])

                y_axis.append(pattern_results[y_var])
                separate_pattern_flag += 1

                # If all the results of the algorithm are collected, plot the results
                if separate_pattern_flag % len(x_axis) == 0:
                    try:
                        ax[varind].scatter(
                            x_axis,
                            y_axis,
                            marker="o",
                            label=CONFIGURATION_PLOT_LABELS[pattern_results["target"]],
                        )
                    except TypeError:
                        ax.scatter(
                            x_axis,
                            y_axis,
                            marker="o",
                            label=CONFIGURATION_PLOT_LABELS[pattern_results["target"]],
                        )
                    y_axis = []
            try:
                ax[varind].set_xlabel(x_label)
                ax[varind].set_ylabel(y_var.capitalize())
                if title is not None:
                    ax[varind].set_title(f"{title} - {y_var.capitalize()}")
                ax[varind].legend(loc="best")
            except TypeError:
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_var.capitalize())
                if title is not None:
                    ax.set_title(f"{title} - {y_var.capitalize()}")
                ax.legend(loc="best")

        if save:
            plt.savefig(f"./figs/{savename}")


# Set up the algorithms, target configurations, and system sizes
class Benchmarking:
    """
    An environment for studying the performance of rearrangement algorithms.

    Can be used to compare the scaling behavior of different algorithms, compare the time it takes for a single algorithm to prepare different target configurations, etc.

    ## Parameters
    - `algos` (list of `Algorithm` objects):
        the algorithms to compare.
    - `figure_output` (`BenchmarkingFigure`):
        an object for plotting.
    - `target_configs` (list of `Configurations` objects OR a list of np.ndarrays representing the explicit target configs.):
        the target patterns to prepare.
        IF a list of np.ndarrays, must provide targets for all system sizes; i.e. must have shape (len(sys_sizes), #targets), where #targets is the number of target configs.
    - `sys_sizes` (range):
        lengths of the square arrays that you want to look at (sqrt(N), where N is the number of tweezer sites).
    - `exp_params` (`PhysicalParams`):
        error and experimental parameters.
    - `n_shots` (int, default 100):
        number of repetitions per (algorithm or target config) per system size.
    - `n_species` (int, default 1):
        number of atomic species.
    - `check_sufficient_atoms` (bool, default True):
        if True, checks whether initial configurations have enough atoms, and regenerates new ones if not.

    ## Example Usage

    Creates an instance of the class and runs a benchmarking round.
        `instance = Benchmarking()`
        `instance.run()`
    """

    def __init__(
        self,
        algos: list = [Algorithm()],
        target_configs: Union[list, np.ndarray] = [Configurations.MIDDLE_FILL],
        error_models_list: list = [ZeroNoise()],
        phys_params_list: list = [PhysicalParams()],
        sys_sizes: list = list(range(10, 16)),
        rounds_list: list = [1],
        figure_output: BenchmarkingFigure = BenchmarkingFigure(),
        n_shots: int = 100,
        n_species: int = 1,
        check_sufficient_atoms: bool = True,
        show_progress: bool = True,
        per_round_logging: bool = False,
    ):
        # initializing the sweep modules (minus target configs, see below)
        self.algos, self.n_algos = algos, len(algos)
        self.system_size_range, self.n_sizes = sys_sizes, len(sys_sizes)
        self.error_models_list, self.n_models = error_models_list, len(
            error_models_list
        )
        self.phys_params_list, self.n_parsets = phys_params_list, len(phys_params_list)
        self.rounds_list, self.n_rounds = rounds_list, len(rounds_list)

        # initializing other variables
        self.n_shots = n_shots
        self.check_sufficient_atoms = check_sufficient_atoms
        self.figure_output = figure_output
        self.show_progress = show_progress
        self.tweezer_array = AtomArray(n_species=n_species)
        self._target_cache: dict[tuple, np.ndarray] = {}
        # Optional per-round logging of empty-site counts and new fills per shot
        self.per_round_logging = bool(per_round_logging)
        self._per_round_records: list[dict] = []

        # initializing target configs depending on whether they were explicitly specified
        if isinstance(target_configs, list):
            self.istargetlist = True
            self.target_configs, self.n_targets = target_configs, len(target_configs)
        elif isinstance(target_configs, np.ndarray):
            self.istargetlist = False
            self.target_configs = target_configs
            self.n_targets = len(target_configs[0])
            if len(target_configs) != self.n_sizes:
                raise IndexError(
                    f"Number of system sizes {self.n_sizes} and shape of `target_configs` {np.shape(target_configs)} does not match. `target_configs` must have shape (len(sys_sizes), [number of target configs]). "
                )
        else:
            raise TypeError(
                "`target_configs` must be a list of Configuration objects or an np.ndarray."
            )

    def save(self, savename):
        if savename[-3:] == ".nc":
            savename = savename[0:-3]
        self.benchmarking_results.to_netcdf(f"data/{savename}.nc")
        print(f"Benchmarking object saved to `data/{savename}.nc`")

    def load(self, loadname):
        if loadname[-3:] == ".nc":
            loadname = loadname[0:-3]
        path = Path(f"data/{loadname}.nc")
        # Let xarray pick an available engine; avoid requiring netcdf4 at runtime
        self.benchmarking_results = xr.open_dataset(path)
        print(f"Data from `data/{loadname}.nc` loaded to `self.benchmarking_results`.")

    def load_params_from_dataset(self, dataset: xr.Dataset):
        """
        Overwrites current parameters for benchmarking sweeps with those
        from another xarray.Dataset object (e.g. `self.benchmarking_results`)

        Useful when wanting to retake data or play around with slightly different parameters.
        Also useful in recreating the figures from the atommovr paper.
        """
        self.algos = dataset["algorithm"].values
        self.target_configs = dataset["target"].values
        self.istargetlist = True
        if isinstance(self.target_configs[0], np.ndarray):
            self.istargetlist = False
        self.system_size_range = dataset["sys size"].values
        self.error_models_list = dataset["error model"].values
        self.phys_params_list = dataset["physical params"].values
        rounds_list = dataset["num rounds"].values
        self.rounds_list = []
        for round in rounds_list:
            self.rounds_list.append(int(round))
        # Prefer explicit attribute if available (written by `run()`)
        n_shots_attr = dataset.attrs.get("n_shots")
        if n_shots_attr is not None:
            self.n_shots = int(n_shots_attr)
        else:
            # Fallback: leave as-is or attempt to infer; keep existing value if uncertain
            try:
                self.n_shots = int(dataset.attrs.get("n_shots", self.n_shots))
            except Exception:
                pass

    def set_observables(self, observables: list):
        # Recreate figure_output to ensure internal mappings are consistent
        current_type = getattr(self.figure_output, "figure_type", "scale")
        self.figure_output = BenchmarkingFigure(
            variables=observables, figure_type=current_type
        )

    @staticmethod
    def _format_target_label(target_entry, index: int) -> str:
        if hasattr(target_entry, "name"):
            return str(target_entry.name)
        if isinstance(target_entry, np.ndarray):
            return f"Custom{index}"
        return str(target_entry)

    @staticmethod
    def _safe_filename_component(name: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
        cleaned = cleaned.strip("_")
        return cleaned or "algorithm"

    @staticmethod
    def _physical_param_repr(parset) -> str:
        try:
            return repr(parset)
        except Exception:
            return str(parset)

    def _target_cache_key(
        self, pattern, base_size: int, occupation_prob: float | None
    ) -> tuple:
        pattern_key = getattr(pattern, "value", pattern)
        occ = 0.0 if occupation_prob is None else float(occupation_prob)
        return (
            pattern_key,
            int(base_size),
            round(occ, 6),
            int(self.tweezer_array.n_species),
        )

    def _get_canonical_target(
        self, pattern, base_size: int, occupation_prob: float | None
    ) -> np.ndarray:
        key = self._target_cache_key(pattern, base_size, occupation_prob)
        cached = self._target_cache.get(key)
        if cached is not None:
            return cached
        base_array = AtomArray(
            [base_size, base_size],
            n_species=self.tweezer_array.n_species,
            params=self.tweezer_array.params,
            error_model=self.tweezer_array.error_model,
        )
        base_array.generate_target(pattern, occupation_prob=occupation_prob)
        target = base_array.target.copy()
        self._target_cache[key] = target
        return target

    @staticmethod
    def _ensure_3d(target: np.ndarray) -> np.ndarray:
        if target.ndim == 3:
            return target
        return target[:, :, np.newaxis]

    def _embed_target(
        self, base_target: np.ndarray, rows: int, cols: int
    ) -> np.ndarray:
        target = self._ensure_3d(base_target)
        base_rows, base_cols, species = target.shape
        if rows < base_rows or cols < base_cols:
            raise ValueError(
                f"Algorithm shape {rows}x{cols} cannot host canonical target {base_rows}x{base_cols}."
            )
        row_start = (rows - base_rows) // 2
        col_start = (cols - base_cols) // 2
        embedded = np.zeros((rows, cols, species), dtype=target.dtype)
        row_slice = slice(row_start, row_start + base_rows)
        col_slice = slice(col_start, col_start + base_cols)
        embedded[row_slice, col_slice, :] = target
        return embedded

    def _prepare_random_targets(
        self, base_size: int, occupation_prob: float | None
    ) -> list[np.ndarray]:
        prob = float(occupation_prob) if occupation_prob is not None else 0.5
        shape = [base_size, base_size]
        return generate_random_target_configs(
            self.n_shots, targ_occup_prob=prob, shape=shape
        )

    def _get_algorithm_shape(
        self, algorithm, base_size: int, loading_prob: float | None = None
    ) -> tuple[int, int]:
        """Return (rows, cols) for the array shape to use with the given algorithm."""
        if not self.istargetlist:
            return base_size, base_size

        geometry_spec = getattr(algorithm, "preferred_geometry_spec", None)
        prob = loading_prob if loading_prob is not None else 0.6
        rows, cols = array_shape_for_geometry(
            geometry_spec, base_size, loading_prob=prob
        )
        rows = max(rows, base_size)
        cols = max(cols, base_size, rows)
        return rows, cols

    def get_result_array_dims(self):
        """
        Updates the size and shape of the storage array
        based on the current set of parameters.
        """
        self.n_algos = len(self.algos)
        if self.istargetlist:
            self.n_targets = len(self.target_configs)
        else:
            self.n_targets = len(self.target_configs[0])
        if isinstance(self.target_configs, list) or not isinstance(
            self.target_configs[0], np.ndarray
        ):
            self.istargetlist = True
            self.n_targets = len(self.target_configs)
        elif isinstance(self.target_configs, np.ndarray):
            self.istargetlist = False
            self.n_targets = len(self.target_configs[0])
            if len(self.target_configs) != self.n_sizes:
                raise IndexError(
                    f"Number of system sizes {self.n_sizes} and shape of `target_configs` {np.shape(self.self.target_configs)} does not match. `target_configs` must have shape (len(sys_sizes), [number of target configs]). "
                )
        else:
            raise TypeError(
                "`target_configs` must be a list of Configuration objects or an np.ndarray."
            )
        self.n_sizes = len(self.system_size_range)
        self.n_models = len(self.error_models_list)
        self.n_parsets = len(self.phys_params_list)
        self.n_rounds = len(self.rounds_list)

    def run(self, do_ejection: bool = False):
        """
        Run a round of benchmarking according to the parameters passed to the `Benchmarking()` object.

        Saves the results in the variable `self.benchmarking_results`.
        """

        # initializing result arrays
        self.get_result_array_dims()
        result_array_dims = [
            self.n_algos,
            self.n_targets,
            self.n_sizes,
            self.n_models,
            self.n_parsets,
            self.n_rounds,
        ]
        success_rate_array = np.zeros(result_array_dims, dtype="float")
        time_array = np.zeros(result_array_dims, dtype="float")
        # Store scalar summaries to ensure NetCDF compatibility
        fill_fracs_array = np.zeros(result_array_dims, dtype="float")
        wrong_places_array = np.zeros(result_array_dims, dtype="float")
        n_atoms_array = np.zeros(result_array_dims, dtype="float")
        n_targets_array = np.zeros(result_array_dims, dtype="float")
        sufficient_atom_rate = np.zeros(result_array_dims, dtype="float")
        wall_time_array = np.zeros(result_array_dims, dtype="float")
        cpu_time_array = np.zeros(result_array_dims, dtype="float")
        move_counts_array = np.zeros(result_array_dims, dtype="float")
        parallel_batch_counts_array = np.zeros(result_array_dims, dtype="float")
        array_rows_array = np.zeros(result_array_dims, dtype="int64")
        array_cols_array = np.zeros(result_array_dims, dtype="int64")
        zstar_array = np.zeros(result_array_dims, dtype="float")

        self._benchmark_records = []
        self._current_run_timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        # for xarray object
        dims = (
            "algorithm",
            "target",
            "sys size",
            "error model",
            "physical params",
            "num rounds",
        )
        if self.istargetlist:
            coord_targets = self.target_configs
        else:
            coord_targets = [f"Custom{i}" for i in range(self.n_targets)]
        # Use string labels for coords to ensure NetCDF serialization compatibility
        algo_labels = [
            getattr(a, "__class__", type(a)).__name__ if not isinstance(a, str) else a
            for a in self.algos
        ]
        err_labels = [
            getattr(e, "__class__", type(e)).__name__ if not isinstance(e, str) else e
            for e in self.error_models_list
        ]

        # Represent physical params succinctly; fallback to class name if repr is too long
        def _param_label(p):
            try:
                s = repr(p)
                return s if len(s) <= 80 else getattr(p, "__class__", type(p)).__name__
            except Exception:
                return getattr(p, "__class__", type(p)).__name__

        phys_labels = [_param_label(p) for p in self.phys_params_list]
        phys_reprs = [self._physical_param_repr(p) for p in self.phys_params_list]

        coords = {
            "algorithm": algo_labels,
            "target": coord_targets,
            "sys size": list(self.system_size_range),
            "error model": err_labels,
            "physical params": phys_labels,
            "num rounds": list(self.rounds_list),
        }

        progress_bars = None
        progress_states = None
        if self.show_progress and tqdm_auto is not None:
            progress_bars = {}
            progress_states = {}
            for algo_label in algo_labels:
                progress_bars[algo_label] = tqdm_auto(
                    total=self.n_sizes,
                    desc=f"{algo_label}",
                    unit="size",
                    leave=False,
                )
                progress_states[algo_label] = set()
        elif self.show_progress and tqdm_auto is None:
            print("[INFO] tqdm is not installed; progress bars disabled.")

        # iterating through sweep parameters and running benchmarking rounds
        for param_ind, parset in enumerate(self.phys_params_list):
            self.tweezer_array.params = parset
            parset_label = phys_labels[param_ind]
            parset_repr = phys_reprs[param_ind]
            loading_prob = getattr(parset, "loading_prob", None)
            target_prob = getattr(parset, "target_occup_prob", None)
            max_rows = int(np.max(self.system_size_range))
            if self.istargetlist:
                storage_cols = max(
                    self._get_algorithm_shape(algo, size, loading_prob=loading_prob)[1]
                    for algo in self.algos
                    for size in self.system_size_range
                )
            else:
                storage_cols = max_rows
            storage_cols = max(storage_cols, max_rows)
            storage_shape = [max_rows, storage_cols]

            self.init_config_storage = generate_random_init_configs(
                self.n_shots,
                load_prob=self.tweezer_array.params.loading_prob,
                shape=storage_shape,
                n_species=self.tweezer_array.n_species,
            )
            for targ_ind in range(self.n_targets):
                pattern = self.target_configs[targ_ind] if self.istargetlist else None
                pattern_enum = pattern if isinstance(pattern, Configurations) else None
                target_label = self._format_target_label(
                    coord_targets[targ_ind], targ_ind
                )
                for model_ind, error_model in enumerate(self.error_models_list):
                    self.tweezer_array.error_model = error_model
                    error_label = err_labels[model_ind]
                    error_repr = repr(error_model)

                    for size_ind, size in enumerate(self.system_size_range):
                        canonical_target = None
                        random_targets = None
                        if self.istargetlist and pattern_enum is not None:
                            if pattern_enum == Configurations.RANDOM:
                                random_prob = getattr(
                                    self.tweezer_array.params, "target_occup_prob", None
                                )
                                if random_prob is None:
                                    random_prob = loading_prob
                                random_targets = self._prepare_random_targets(
                                    size, random_prob
                                )
                            else:
                                canonical_prob = loading_prob
                                if canonical_prob is None:
                                    canonical_prob = getattr(
                                        self.tweezer_array.params,
                                        "target_occup_prob",
                                        None,
                                    )
                                canonical_target = self._get_canonical_target(
                                    pattern_enum, size, canonical_prob
                                )

                        for alg_ind, algo in enumerate(self.algos):
                            rows, cols = self._get_algorithm_shape(
                                algo, size, loading_prob=loading_prob
                            )
                            precomputed_target = None
                            if not self.istargetlist:
                                rows = size
                                cols = size
                                self.tweezer_array.target = self.target_configs[
                                    size_ind, targ_ind
                                ]
                            else:
                                if (
                                    pattern_enum != Configurations.RANDOM
                                    and canonical_target is not None
                                ):
                                    precomputed_target = self._embed_target(
                                        canonical_target, rows, cols
                                    )

                            self.tweezer_array.shape = [rows, cols]
                            algo_label = algo_labels[alg_ind]
                            for round_ind, num_rounds in enumerate(self.rounds_list):
                                (
                                    success_rate,
                                    mean_success_time,
                                    fill_fracs,
                                    wrong_places,
                                    atoms_in_arrays,
                                    atoms_in_target,
                                    sufficient_rate,
                                ) = self._run_benchmark_round(
                                    algo,
                                    do_ejection=do_ejection,
                                    pattern=pattern,
                                    num_rounds=num_rounds,
                                    precomputed_target=precomputed_target,
                                    random_targets=random_targets,
                                    base_target_size=size,
                                )
                                # Read any extra metrics produced by the round (backwards-compatible storage)
                                extra = getattr(self, "_last_round_extra", {})
                                wall_time = float(extra.get("wall_elapsed", np.nan))
                                cpu_time = float(extra.get("cpu_elapsed", np.nan))
                                parallel_counts = extra.get("parallel_move_counts", [])
                                move_counts = extra.get("atom_move_counts", [])
                                per_round_new_fills = extra.get(
                                    "per_round_new_fills", []
                                )
                                per_round_empty_counts = extra.get(
                                    "per_round_empty_counts", []
                                )
                                zstar_vals = extra.get("zstar_values", [])
                                fill_mean = (
                                    float(np.mean(fill_fracs))
                                    if len(fill_fracs) > 0
                                    else np.nan
                                )
                                wrong_mean = (
                                    float(np.mean(wrong_places))
                                    if len(wrong_places) > 0
                                    else np.nan
                                )
                                atoms_mean = (
                                    float(np.mean(atoms_in_arrays))
                                    if len(atoms_in_arrays) > 0
                                    else np.nan
                                )
                                target_mean = (
                                    float(np.mean(atoms_in_target))
                                    if len(atoms_in_target) > 0
                                    else np.nan
                                )
                                parallel_mean = (
                                    float(np.mean(parallel_counts))
                                    if len(parallel_counts) > 0
                                    else np.nan
                                )
                                moves_mean = (
                                    float(np.mean(move_counts))
                                    if len(move_counts) > 0
                                    else np.nan
                                )
                                zstar_mean = (
                                    float(np.nanmean(zstar_vals))
                                    if len(zstar_vals) > 0
                                    else np.nan
                                )
                                # populating result arrays
                                success_rate_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = success_rate
                                time_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = mean_success_time
                                # Reduce per-shot lists to means for storage
                                fill_fracs_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = fill_mean
                                wrong_places_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = wrong_mean
                                n_atoms_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = atoms_mean
                                n_targets_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = target_mean
                                sufficient_atom_rate[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = sufficient_rate
                                wall_time_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = wall_time
                                cpu_time_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = cpu_time
                                parallel_batch_counts_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = parallel_mean
                                move_counts_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = moves_mean
                                array_rows_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = rows
                                array_cols_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = cols
                                zstar_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = zstar_mean

                                record = {
                                    "run_timestamp": self._current_run_timestamp,
                                    "algorithm": algo_label,
                                    "target": target_label,
                                    "sys_size": size,
                                    "array_rows": rows,
                                    "array_cols": cols,
                                    "num_rounds": num_rounds,
                                    "round_index": round_ind,
                                    "error_model": error_label,
                                    "error_model_repr": error_repr,
                                    "physical_params_label": parset_label,
                                    "physical_params_repr": parset_repr,
                                    "loading_prob": loading_prob,
                                    "target_occup_prob": target_prob,
                                    "success_rate": success_rate,
                                    "mean_success_time": mean_success_time,
                                    "wall_time_seconds": wall_time,
                                    "cpu_time_seconds": cpu_time,
                                    "mean_filling_fraction": fill_mean,
                                    "mean_wrong_places": wrong_mean,
                                    "mean_atoms_in_array": atoms_mean,
                                    "mean_atoms_in_target": target_mean,
                                    "sufficient_atom_rate": sufficient_rate,
                                    "mean_moves_per_shot": moves_mean,
                                    "mean_parallel_batches_per_shot": parallel_mean,
                                    "mean_zstar": zstar_mean,
                                    "n_shots": self.n_shots,
                                    "n_species": self.tweezer_array.n_species,
                                    "check_sufficient_atoms": self.check_sufficient_atoms,
                                    "do_ejection": do_ejection,
                                }
                                self._benchmark_records.append(record)
                                # Save per-round CSV if requested
                                if self.per_round_logging and per_round_new_fills:
                                    base_dir = Path("data/benchmark_pipeline/per_round")
                                    base_dir.mkdir(parents=True, exist_ok=True)
                                    fname = f"per_round_{self._current_run_timestamp}_{algo_label}_L{size}_{error_label}_R{num_rounds}_{target_label}.csv"
                                    file_path = base_dir / fname
                                    with open(
                                        file_path, "w", newline="", encoding="utf-8"
                                    ) as csv_file:
                                        fieldnames = [
                                            "shot",
                                            "round_index",
                                            "empty_before",
                                            "new_fills",
                                        ]
                                        writer = csv.DictWriter(
                                            csv_file, fieldnames=fieldnames
                                        )
                                        writer.writeheader()
                                        for shot_idx, (
                                            fills_list,
                                            empty_list,
                                        ) in enumerate(
                                            zip(
                                                per_round_new_fills,
                                                per_round_empty_counts,
                                            )
                                        ):
                                            for r_idx, (nf, eb) in enumerate(
                                                zip(fills_list, empty_list)
                                            ):
                                                writer.writerow(
                                                    {
                                                        "shot": shot_idx,
                                                        "round_index": r_idx,
                                                        "empty_before": int(eb),
                                                        "new_fills": int(nf),
                                                    }
                                                )
                                if progress_bars:
                                    bar = progress_bars[algo_label]
                                    bar.set_postfix({"L": size})
                                    if size not in progress_states[algo_label]:
                                        progress_states[algo_label].add(size)
                                        bar.update(1)

        success_rates_da = xr.DataArray(success_rate_array, dims=dims, coords=coords)
        success_times_da = xr.DataArray(time_array, dims=dims, coords=coords)
        fill_fracs_da = xr.DataArray(fill_fracs_array, dims=dims, coords=coords)
        wrong_places_da = xr.DataArray(wrong_places_array, dims=dims, coords=coords)
        n_atoms_da = xr.DataArray(n_atoms_array, dims=dims, coords=coords)
        n_targets_da = xr.DataArray(n_targets_array, dims=dims, coords=coords)
        sufficient_atom_rate_da = xr.DataArray(
            sufficient_atom_rate, dims=dims, coords=coords
        )
        wall_time_da = xr.DataArray(wall_time_array, dims=dims, coords=coords)
        cpu_time_da = xr.DataArray(cpu_time_array, dims=dims, coords=coords)
        move_counts_da = xr.DataArray(move_counts_array, dims=dims, coords=coords)
        parallel_batches_da = xr.DataArray(
            parallel_batch_counts_array, dims=dims, coords=coords
        )
        array_rows_da = xr.DataArray(array_rows_array, dims=dims, coords=coords)
        array_cols_da = xr.DataArray(array_cols_array, dims=dims, coords=coords)
        zstar_da = xr.DataArray(zstar_array, dims=dims, coords=coords)

        self.benchmarking_results = xr.Dataset(
            {
                "success rate": success_rates_da,
                "time": success_times_da,
                "filling fraction": fill_fracs_da,
                "wrong places": wrong_places_da,
                "n atoms": n_atoms_da,
                "n targets": n_targets_da,
                "sufficient rate": sufficient_atom_rate_da,
                "wall time": wall_time_da,
                "cpu time": cpu_time_da,
                "mean moves": move_counts_da,
                "parallel move batches": parallel_batches_da,
                "array rows": array_rows_da,
                "array cols": array_cols_da,
                "zstar lower bound": zstar_da,
            }
        )
        self.benchmarking_results.attrs.update(
            {
                "do_ejection": bool(do_ejection),
                "n_shots": int(self.n_shots),
                "n_species": int(self.tweezer_array.n_species),
            }
        )

        self._summarize_complexity()
        self._export_runtime_csv()
        self._export_complexity_csv()
        self._export_algorithm_summary_csv()
        self._compute_analytical_models()

        if progress_bars:
            for bar in progress_bars.values():
                bar.close()

    def _export_runtime_csv(self):
        records = getattr(self, "_benchmark_records", None)
        if not records:
            return

        run_id = getattr(
            self,
            "_current_run_timestamp",
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        base_dir = Path("data/benchmark_pipeline/runtime_exports")
        base_dir.mkdir(parents=True, exist_ok=True)

        field_order = [
            "run_timestamp",
            "algorithm",
            "target",
            "sys_size",
            "array_rows",
            "array_cols",
            "num_rounds",
            "round_index",
            "error_model",
            "error_model_repr",
            "physical_params_label",
            "physical_params_repr",
            "loading_prob",
            "target_occup_prob",
            "success_rate",
            "mean_success_time",
            "wall_time_seconds",
            "mean_filling_fraction",
            "mean_wrong_places",
            "mean_atoms_in_array",
            "mean_atoms_in_target",
            "sufficient_atom_rate",
            "mean_moves_per_shot",
            "mean_parallel_batches_per_shot",
            "mean_zstar",
            "n_shots",
            "n_species",
            "check_sufficient_atoms",
            "do_ejection",
            "cpu_time_seconds",
        ]

        grouped_records = {}
        for rec in records:
            grouped_records.setdefault(rec["algorithm"], []).append(rec)

        for algo, rows in grouped_records.items():
            safe_name = self._safe_filename_component(algo)
            file_path = base_dir / f"{safe_name}_benchmark_runtime_{run_id}.csv"
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=field_order)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        # retain exported data for callers if needed
        self._benchmark_records = records

    def _summarize_complexity(self):
        records = getattr(self, "_benchmark_records", None)
        summary = []
        if not records:
            self._complexity_summary = summary
            return

        def _positive_float(value):
            try:
                val = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(val) or val <= 0:
                return None
            return val

        def _format_axis_value(value: float) -> str:
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return f"{value:.6g}"

        axis_specs = {
            "system_size": {
                "label": "side length",
                "getter": lambda rec: rec["sys_size"],
            },
            "target_sites": {
                "label": "target sites",
                "getter": lambda rec: rec["mean_atoms_in_target"],
            },
        }

        metric_specs = [
            {
                "name": "wall_time",
                "field": "wall_time_seconds",
                "label": "wall time (s)",
                "units": "seconds",
                "axes": ["system_size", "target_sites"],
            },
            {
                "name": "moves",
                "field": "mean_moves_per_shot",
                "label": "moves per shot",
                "units": "moves",
                "axes": ["target_sites"],
            },
            {
                "name": "parallel_batches",
                "field": "mean_parallel_batches_per_shot",
                "label": "parallel batches per shot",
                "units": "batches",
                "axes": ["target_sites"],
            },
        ]

        by_algorithm = defaultdict(list)
        for rec in records:
            by_algorithm[rec["algorithm"]].append(rec)

        for algo, algo_records in by_algorithm.items():
            combo_groups = defaultdict(list)
            for rec in algo_records:
                combo_key = (
                    rec["target"],
                    rec["error_model"],
                    rec["error_model_repr"],
                    rec["physical_params_label"],
                    rec["physical_params_repr"],
                    rec["num_rounds"],
                    rec["do_ejection"],
                )
                combo_groups[combo_key].append(rec)

            for combo_key, combo_records in combo_groups.items():
                for metric in metric_specs:
                    metric_field = metric["field"]
                    for axis_name in metric["axes"]:
                        axis_spec = axis_specs[axis_name]
                        axis_map: dict[float, list[float]] = defaultdict(list)
                        axis_getter = axis_spec["getter"]
                        for rec in combo_records:
                            axis_val = _positive_float(axis_getter(rec))
                            metric_val = _positive_float(rec.get(metric_field))
                            if axis_val is None or metric_val is None:
                                continue
                            axis_map[axis_val].append(metric_val)
                        if not axis_map:
                            continue
                        axis_values_sorted = sorted(axis_map.keys())
                        axis_vals = np.array(axis_values_sorted, dtype=float)
                        mean_metrics = np.array(
                            [np.mean(axis_map[val]) for val in axis_values_sorted],
                            dtype=float,
                        )
                        mask = (axis_vals > 0) & (mean_metrics > 0)
                        if np.count_nonzero(mask) < 2:
                            continue
                        axis_subset = axis_vals[mask]
                        metric_subset = mean_metrics[mask]
                        x = np.log(axis_subset)
                        y = np.log(metric_subset)
                        slope, intercept = np.polyfit(x, y, 1)
                        y_hat = slope * x + intercept
                        ss_res = float(np.sum((y - y_hat) ** 2))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
                        coef = float(np.exp(intercept))
                        combo_sample = combo_records[0]
                        axis_tokens = [_format_axis_value(val) for val in axis_subset]
                        metric_tokens = [f"{val:.6g}" for val in metric_subset]
                        summary_record = {
                            "run_timestamp": combo_sample["run_timestamp"],
                            "algorithm": algo,
                            "target": combo_key[0],
                            "error_model": combo_key[1],
                            "error_model_repr": combo_key[2],
                            "physical_params_label": combo_key[3],
                            "physical_params_repr": combo_key[4],
                            "num_rounds": combo_key[5],
                            "do_ejection": combo_key[6],
                            "loading_prob": combo_sample["loading_prob"],
                            "target_occup_prob": combo_sample["target_occup_prob"],
                            "n_shots": combo_sample["n_shots"],
                            "n_species": combo_sample["n_species"],
                            "check_sufficient_atoms": combo_sample[
                                "check_sufficient_atoms"
                            ],
                            "scale_axis": axis_name,
                            "axis_label": axis_spec["label"],
                            "metric_name": metric["name"],
                            "metric_label": metric["label"],
                            "metric_units": metric["units"],
                            "exponent": float(slope),
                            "coefficient": coef,
                            "r_squared": float(r_squared),
                            "n_points": int(np.count_nonzero(mask)),
                            "size_min": float(np.min(axis_subset)),
                            "size_max": float(np.max(axis_subset)),
                            "sizes": "|".join(axis_tokens),
                            "metric_values": "|".join(metric_tokens),
                            "mean_wall_times": (
                                "|".join(metric_tokens)
                                if metric["name"] == "wall_time"
                                else ""
                            ),
                        }
                        summary.append(summary_record)
                        print(
                            f"[Complexity] {algo} target={combo_key[0]} rounds={combo_key[5]} "
                            f"error={combo_key[1]} metric={metric['name']} axis={axis_name}: "
                            f"{metric['label']} ≈ {coef:.3g} * L^{slope:.2f} "
                            f"(L represents {axis_spec['label']}, R^2={r_squared:.3f}, values={axis_tokens})"
                        )

        if not summary:
            print(
                "[Complexity] Not enough distinct inputs to estimate runtime scaling."
            )
        self._complexity_summary = summary

    def _export_complexity_csv(self):
        summary = getattr(self, "_complexity_summary", None)
        if not summary:
            return

        run_id = getattr(
            self,
            "_current_run_timestamp",
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        base_dir = Path("data/benchmark_pipeline/runtime_exports")
        base_dir.mkdir(parents=True, exist_ok=True)

        field_order = [
            "run_timestamp",
            "algorithm",
            "target",
            "error_model",
            "error_model_repr",
            "physical_params_label",
            "physical_params_repr",
            "num_rounds",
            "do_ejection",
            "loading_prob",
            "target_occup_prob",
            "n_shots",
            "n_species",
            "check_sufficient_atoms",
            "scale_axis",
            "axis_label",
            "metric_name",
            "metric_label",
            "metric_units",
            "exponent",
            "coefficient",
            "r_squared",
            "n_points",
            "size_min",
            "size_max",
            "sizes",
            "metric_values",
            "mean_wall_times",
        ]

        grouped = defaultdict(list)
        for rec in summary:
            grouped[rec["algorithm"]].append(rec)

        for algo, rows in grouped.items():
            safe_name = self._safe_filename_component(algo)
            file_path = base_dir / f"{safe_name}_complexity_{run_id}.csv"
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=field_order)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

    def _export_algorithm_summary_csv(self):
        """Write a single CSV summarizing average metrics per algorithm.

        Uses self._benchmark_records (populated by `run`) and writes
        `data/benchmark_pipeline/summary_exports/algorithm_summary_{run_id}.csv`.
        """
        records = getattr(self, "_benchmark_records", None)
        if not records:
            return

        run_id = getattr(
            self,
            "_current_run_timestamp",
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        base_dir = Path("data/benchmark_pipeline/summary_exports")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Group by algorithm and compute simple means for numeric fields
        by_algo = defaultdict(list)
        for rec in records:
            by_algo[rec["algorithm"]].append(rec)

        field_order = [
            "run_timestamp",
            "algorithm",
            "n_records",
            "mean_success_rate",
            "mean_success_time",
            "mean_wall_time_seconds",
            "mean_filling_fraction",
            "mean_wrong_places",
            "mean_atoms_in_array",
            "mean_atoms_in_target",
            "mean_moves_per_shot",
            "mean_parallel_batches_per_shot",
            "mean_sufficient_rate",
            "loading_prob",
            "target_occup_prob",
            "n_shots",
            "n_species",
        ]

        def _safe_mean(lst, key):
            vals = [float(r[key]) for r in lst if key in r and r[key] is not None]
            if not vals:
                return ""
            return float(np.mean(vals))

        rows_out = []
        for algo, recs in by_algo.items():
            row = {
                "run_timestamp": run_id,
                "algorithm": algo,
                "n_records": len(recs),
                "mean_success_rate": _safe_mean(recs, "success_rate"),
                "mean_success_time": _safe_mean(recs, "mean_success_time"),
                "mean_wall_time_seconds": _safe_mean(recs, "wall_time_seconds"),
                "mean_filling_fraction": _safe_mean(recs, "mean_filling_fraction"),
                "mean_wrong_places": _safe_mean(recs, "mean_wrong_places"),
                "mean_atoms_in_array": _safe_mean(recs, "mean_atoms_in_array"),
                "mean_atoms_in_target": _safe_mean(recs, "mean_atoms_in_target"),
                "mean_moves_per_shot": _safe_mean(recs, "mean_moves_per_shot"),
                "mean_parallel_batches_per_shot": _safe_mean(
                    recs, "mean_parallel_batches_per_shot"
                ),
                "mean_sufficient_rate": _safe_mean(recs, "sufficient_atom_rate"),
                "loading_prob": recs[0].get("loading_prob", ""),
                "target_occup_prob": recs[0].get("target_occup_prob", ""),
                "n_shots": recs[0].get("n_shots", ""),
                "n_species": recs[0].get("n_species", ""),
            }
            rows_out.append(row)

        file_path = base_dir / f"algorithm_summary_{run_id}.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_order)
            writer.writeheader()
            for row in rows_out:
                writer.writerow(row)

        # retain for callers/tests
        self._algorithm_summary = rows_out

    def _compute_analytical_models(self):
        """Derive analytical success rate models P(N) = r(N)^N from filling fractions.

        For each (algorithm, error_model) pair the mean filling fraction at each
        measured system size serves as an empirical per-site fill probability r.
        Interpolating log(r) across target-site counts and exponentiating by N
        yields a smooth analytical prediction that can be overlaid on success-rate
        scatter plots.
        """
        ds = getattr(self, "benchmarking_results", None)
        if ds is None:
            self._analytical_models = {}
            return

        models = {}
        algo_labels = list(ds.coords["algorithm"].values)
        err_labels = list(ds.coords["error model"].values)
        target_sel = ds.coords["target"].values[0]
        phys_sel = ds.coords["physical params"].values[0]
        round_sel = ds.coords["num rounds"].values[0]

        for algo in algo_labels:
            for err in err_labels:
                base_sel = {
                    "algorithm": algo,
                    "target": target_sel,
                    "error model": err,
                    "physical params": phys_sel,
                    "num rounds": round_sel,
                }
                fill_fracs = ds["filling fraction"].sel(**base_sel).values.flatten()
                n_targets = ds["n targets"].sel(**base_sel).values.flatten()

                mask = (
                    np.isfinite(fill_fracs)
                    & np.isfinite(n_targets)
                    & (n_targets > 0)
                    & (fill_fracs > 0)
                )
                if np.count_nonzero(mask) < 2:
                    continue

                n_arr = n_targets[mask]
                r_arr = np.clip(fill_fracs[mask], 1e-10, 1.0)

                # aggregate per system size: mean filling fraction per unique N
                unique_n = np.unique(n_arr)
                mean_r = np.array([r_arr[n_arr == n].mean() for n in unique_n])
                if len(unique_n) < 2:
                    continue
                log_r = np.log(mean_r)

                def _make_model(n_pts, log_r_pts):
                    def model(N):
                        N = np.asarray(N, dtype=float)
                        log_r_interp = np.interp(
                            N,
                            n_pts,
                            log_r_pts,
                            left=log_r_pts[0],
                            right=log_r_pts[-1],
                        )
                        return np.exp(N * log_r_interp)

                    return model

                models[(str(algo), str(err))] = _make_model(
                    unique_n.copy(),
                    log_r.copy(),
                )

        self._analytical_models = models

    def _run_benchmark_round(
        self,
        algorithm,
        do_ejection: bool = False,
        pattern=None,
        num_rounds=1,
        precomputed_target: np.ndarray | None = None,
        random_targets: list[np.ndarray] | None = None,
        base_target_size: int | None = None,
    ) -> tuple[float, float, list, list, list, list, float]:
        success_times = []
        success_flags = []
        filling_fractions = []
        wrong_places = []
        atoms_in_arrays = []
        atoms_in_targets = []
        sufficient_flags = []
        parallel_move_counts = []
        atom_move_counts = []
        zstar_values = []
        # Per-round diagnostics
        per_round_new_fills_allshots: list[list[int]] = []
        per_round_empty_counts_allshots: list[list[int]] = []

        pattern_enum = pattern if isinstance(pattern, Configurations) else None

        if (
            self.istargetlist
            and precomputed_target is None
            and pattern_enum not in (None, Configurations.RANDOM)
        ):
            self.tweezer_array.generate_target(
                pattern, occupation_prob=self.tweezer_array.params.loading_prob
            )

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        for shot in range(self.n_shots):
            # getting initial and final target configs
            raw_init = np.asarray(self.init_config_storage[shot])
            rows_needed = int(self.tweezer_array.shape[0])
            cols_needed = int(self.tweezer_array.shape[1])
            n_species = int(self.tweezer_array.n_species)

            # Normalize raw_init to 2D or 3D array matching species
            if raw_init.ndim == 3 and raw_init.shape[2] == n_species:
                src = raw_init
            elif raw_init.ndim == 2 and n_species == 1:
                src = raw_init
            elif raw_init.ndim == 3 and raw_init.shape[2] != n_species:
                # If species mismatch, take first species plane as source
                src = raw_init[..., 0]
            else:
                src = raw_init

            # Prepare a matrix of the required shape and copy/crop or pad as needed
            if src.ndim == 2:
                init_matrix = np.zeros((rows_needed, cols_needed), dtype=src.dtype)
                r = min(src.shape[0], rows_needed)
                c = min(src.shape[1], cols_needed)
                init_matrix[:r, :c] = src[:r, :c]
                self.tweezer_array.matrix = init_matrix.reshape(
                    [rows_needed, cols_needed, 1]
                )
            else:
                init_matrix = np.zeros(
                    (rows_needed, cols_needed, n_species), dtype=src.dtype
                )
                r = min(src.shape[0], rows_needed)
                c = min(src.shape[1], cols_needed)
                s = min(src.shape[2], n_species)
                init_matrix[:r, :c, :s] = src[:r, :c, :s]
                self.tweezer_array.matrix = init_matrix

            # provide a local  reference to the initial configuration used for checks
            initial_config = self.tweezer_array.matrix
            if self.istargetlist:
                if precomputed_target is not None and pattern_enum not in (
                    None,
                    Configurations.RANDOM,
                ):
                    self.tweezer_array.target = precomputed_target.copy()
                elif pattern_enum == Configurations.RANDOM:
                    if random_targets is None:
                        base_rows = (
                            base_target_size
                            if base_target_size is not None
                            else self.tweezer_array.shape[0]
                        )
                        random_targets = self._prepare_random_targets(
                            base_rows,
                            getattr(
                                self.tweezer_array.params, "target_occup_prob", None
                            ),
                        )
                    base_random = random_targets[shot % len(random_targets)]
                    base_random = self._ensure_3d(base_random)
                    embedded_random = self._embed_target(
                        base_random,
                        self.tweezer_array.shape[0],
                        self.tweezer_array.shape[1],
                    )
                    self.tweezer_array.target = embedded_random
            if self.check_sufficient_atoms:
                # loop to ensure that the initial configuration has sufficient atoms.
                init_count = 0
                while (
                    np.sum(initial_config) < np.sum(self.tweezer_array.target)
                    and init_count < 100
                ):
                    self.tweezer_array.load_tweezers()
                    initial_config = self.tweezer_array.matrix
                    init_count += 1
                if init_count == 100:
                    print(
                        f"[WARNING] could not find initial configuration with enough atoms ({np.sum(self.tweezer_array.target)}) in target). \
                          Consider aborting run and choosing more suitable parameters. If this is intentional, however, you can turn off this check by setting `check_sufficient_atoms` to False when calling `Benchmarking()`."
                    )
            round_count = 0
            if num_rounds <= 0 or not isinstance(num_rounds, int):
                raise ValueError(
                    f"Number of rearrangement rounds (entered as {num_rounds}) cannot be 0, negative, nor a non-integer value."
                )
            shot_parallel_batches = 0
            shot_move_count = 0
            t_total = 0.0
            # Default to failed shot unless a success_flag is set by the algorithm
            success_flag = 0

            # Compute Z* (bottleneck lower bound) before rearrangement
            try:
                zstar_val = calculate_Zstar_better(
                    self.tweezer_array.matrix,
                    self.tweezer_array.target,
                    self.tweezer_array.n_species,
                    metric="grid",
                )
            except Exception:
                zstar_val = np.nan
            zstar_values.append(float(zstar_val))
            # per-shot per-round lists
            shot_new_fills: list[int] = []
            shot_empty_before: list[int] = []
            while round_count < num_rounds:
                # record number of empty target sites before this round
                filled_before = int(
                    np.sum(
                        np.multiply(
                            self.tweezer_array.matrix, self.tweezer_array.target
                        )
                    )
                )
                total_target = int(np.sum(self.tweezer_array.target))
                empty_before = int(max(0, total_target - filled_before))
                shot_empty_before.append(empty_before)
                # generating and evaluating moves
                try:
                    if self.tweezer_array.n_species == 1:
                        _, move_list, algo_success_flag = algorithm.get_moves(
                            self.tweezer_array, do_ejection=do_ejection
                        )
                    else:
                        _, move_list, algo_success_flag = algorithm.get_moves(
                            self.tweezer_array
                        )
                    t_total, move_stats = self.tweezer_array.evaluate_moves(move_list)
                except ValueError as value_error:
                    print(
                        f"ValueError in round {round_count} for algorithm {algorithm.__class__.__name__}: {value_error}. Marking shot as failed."
                    )
                    break
                # compute new fills achieved by this round
                filled_after = int(
                    np.sum(
                        np.multiply(
                            self.tweezer_array.matrix, self.tweezer_array.target
                        )
                    )
                )
                new_fills = max(0, filled_after - filled_before)
                shot_new_fills.append(int(new_fills))
                if isinstance(move_stats, (list, tuple)) and len(move_stats) > 0:
                    shot_parallel_batches += int(move_stats[0])
                    if len(move_stats) > 1:
                        shot_move_count += int(move_stats[1])
                    else:
                        shot_move_count += int(move_stats[0])
                success_flag = Algorithm.get_success_flag(
                    self.tweezer_array.matrix,
                    self.tweezer_array.target,
                    do_ejection=do_ejection,
                    n_species=self.tweezer_array.n_species,
                )
                if success_flag == 1:
                    # pad remaining rounds with zeros for consistent length
                    round_count += 1
                    # If we ended early, remaining rounds contribute no new fills and empty_before becomes 0
                    # (we still want per-round vectors of length num_rounds)
                    while round_count < num_rounds:
                        shot_empty_before.append(0)
                        shot_new_fills.append(0)
                        round_count += 1
                    break
                round_count += 1

            success_flags.append(success_flag)
            if success_flag:
                success_times.append(t_total)
            parallel_move_counts.append(shot_parallel_batches)
            atom_move_counts.append(shot_move_count)

            # store per-shot per-round diagnostics
            per_round_new_fills_allshots.append(shot_new_fills)
            per_round_empty_counts_allshots.append(shot_empty_before)

            # calculate filling fraction
            filling_fraction_config = np.multiply(
                self.tweezer_array.matrix, self.tweezer_array.target
            )
            filling_fractions.append(
                float(
                    np.sum(filling_fraction_config) / np.sum(self.tweezer_array.target)
                )
            )

            # Identify wrong places (atoms that are not in the target configuration)
            if do_ejection:
                wrong_places.append(
                    int(
                        np.sum(
                            np.abs(
                                self.tweezer_array.matrix - self.tweezer_array.target
                            )
                        )
                    )
                )
            else:
                start_row, end_row, start_col, end_col = get_effective_target_grid(
                    self.tweezer_array.target
                )
                wrong_places.append(
                    int(
                        np.sum(
                            np.abs(
                                self.tweezer_array.matrix[
                                    start_row : end_row + 1, start_col : end_col + 1
                                ]
                                - self.tweezer_array.target[
                                    start_row : end_row + 1, start_col : end_col + 1
                                ]
                            )
                        )
                    )
                )
            # Count atoms in array
            atoms_in_arrays.append(int(np.sum(self.tweezer_array.matrix)))
            atoms_in_targets.append(int(np.sum(self.tweezer_array.target)))

            if np.sum(initial_config) < np.sum(self.tweezer_array.target):
                sufficient_flags.append(False)
            else:
                sufficient_flags.append(True)

        wall_elapsed = (time.perf_counter() - wall_start) / self.n_shots
        cpu_elapsed = (time.process_time() - cpu_start) / self.n_shots

        # Store extra diagnostics for callers that need them (keeps API backward-compatible)
        self._last_round_extra = {
            "wall_elapsed": wall_elapsed,
            "cpu_elapsed": cpu_elapsed,
            "parallel_move_counts": parallel_move_counts,
            "atom_move_counts": atom_move_counts,
            "per_round_new_fills": per_round_new_fills_allshots,
            "per_round_empty_counts": per_round_empty_counts_allshots,
            "zstar_values": zstar_values,
        }

        return (
            float(np.mean(success_flags)),
            float(np.mean(success_times)),
            filling_fractions,
            wrong_places,
            atoms_in_arrays,
            atoms_in_targets,
            float(np.mean(sufficient_flags)),
        )

    def plot_results(self, save=False, savename=None):
        """
        NB: This is a placeholder function for future feature development. See BenchmarkingFigure() for more details.
        """
        if self.figure_output.figure_type == "scale":
            if savename is None:
                savename = "scaling"
            if savename == None:
                savename = "scaling"
            self.figure_output.generate_scaling_figure(
                list(self.system_size_range),
                self.benchmarking_results,
                None,
                "Target sites (atoms)",
                savename=savename,
                save=save,
                complexity_summary=getattr(self, "_complexity_summary", None),
                analytical_model_fns=getattr(self, "_analytical_models", None),
            )

        elif self.figure_output.figure_type == "hist":
            if savename is None:
                savename = "histogram"
            self.figure_output.generate_histogram_figure(
                self.benchmarking_results,
                "Benchmarking results",
                "Array length (# atoms)",
                save=save,
                savename=savename,
            )

        elif self.figure_output.figure_type == "pattern":
            if savename is None:
                savename = "pattern"
            self.figure_output.generate_pattern_figure(
                list(self.system_size_range),
                self.benchmarking_results,
                "Benchmarking results",
                "Array length (# atoms)",
            )
