"""
Reproducible AMS + L-moment distribution fitting + return level curves (SSP245)

Fixes applied:
- Removed force_distribution (always auto-select best distribution based on metrics)
- Fixed SSP245/SSP585 title bug (uses scenario everywhere)
- Deterministic bootstraps (seeded RNG)
- Safe failure if no common years across models
- Robust distribution selection when NaNs occur (penalize missing L-moments)
- Removed absolute Windows path (uses relative paths by default)
- Saves key figures + metrics to outputs/
"""

# IMPORTS
from pathlib import Path
from pprint import pprint
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import lmoments3 as lm
from lmoments3 import distr


# USER SETTINGS (EDIT THESE)
scenario = "ssp245"

# Put your input files inside your repo, e.g. data/raw/
# (Change these paths to match your repo structure.)
model_files = {
    "HadGEM3_ssp245": "data/raw/HadGEM3_ssp245.txt",
    "Noresm_ES2L_ssp245": "data/raw/NorESM2_ssp245.txt",
    "MPI_ESM2_ssp245": "data/raw/MPIesm2_ssp245.txt",
}
obs_file = "data/obs/GPM_Daily_obs.txt"

return_periods = [5, 10, 25, 50, 100, 200]
distributions = ["gev","exp", "gum", "wei", "gpa", "pe3", "gam", "glo"]

# Reproducibility
SEED = 42

# Outputs
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"

# Toggle extra plots (not in your screenshots)
PLOT_BAR_CHART = False
PLOT_HEATMAP = False


# FUNCTION DEFINITIONS
def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)


def load_and_transform_daily_data(file_path: str | Path) -> pd.Series | None:
    """
    Reads a daily precipitation file (Date, Prcp), converts Date,
    computes a 3-day rolling sum, and returns the annual maximum (AMS) as a Series.
    Assumes whitespace-delimited with two columns: YYYY-MM-DD  value
    """
    file_path = Path(file_path)
    try:
        data = pd.read_csv(file_path, sep=r"\s+", header=None, names=["Date", "Prcp"])
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d", errors="coerce")
        data["Prcp"] = pd.to_numeric(data["Prcp"], errors="coerce")
        data.dropna(subset=["Date", "Prcp"], inplace=True)

        data.set_index("Date", inplace=True)
        data.sort_index(inplace=True)

        # 3-day rolling sum
        data["rolling_3day"] = data["Prcp"].rolling(window=3).sum()

        # Annual maxima of 3-day sums
        annual_max = data.groupby(data.index.year)["rolling_3day"].max()
        annual_max_df = pd.DataFrame({"Year": annual_max.index, "AMS": annual_max.values})
        annual_max_df.sort_values(by="Year", inplace=True)
        annual_max_df.set_index("Year", inplace=True)

        print(f"Successfully loaded and transformed data from: {file_path}")
        return annual_max_df["AMS"]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def compute_empirical_lmoments(annual_max: pd.Series):
    """Computes empirical L-moments (L-skew=t3 and L-kurtosis=t4)."""
    try:
        lmoments_empirical = lm.lmom_ratios(annual_max.values)
        return lmoments_empirical[2], lmoments_empirical[3]  # t3, t4
    except Exception as e:
        print(f"Error computing empirical L‐moments: {e}")
        return None, None


def fit_distributions(annual_max: pd.Series, distributions_list: list[str]):
    """Fits each distribution (L-moments) and returns dict(dist_name -> fit_params dict)."""
    fit_results = {}
    for dist_name in distributions_list:
        try:
            dist_class = getattr(distr, dist_name.lower())
            fit = dist_class.lmom_fit(annual_max.values)
            fit_results[dist_name] = fit
            print(f"Fitted {dist_name.upper()} successfully.")
        except AttributeError:
            print(f"Distribution {dist_name.upper()} not available in lmoments3.")
        except Exception as e:
            print(f"Error fitting {dist_name.upper()}: {e}")
    return fit_results


def perform_ks_test(data: pd.Series, dist_name: str, fit_params: dict):
    """Kolmogorov–Smirnov test between data and the fitted distribution."""
    try:
        dist_class = getattr(distr, dist_name.lower())

        # NOTE: lmoments3 APIs use positional parameters; we rely on the dict insertion order
        # returned by lmom_fit(). This is typically stable for lmoments3 fits.
        params = list(fit_params.values())

        cdf_func = lambda x: dist_class.cdf(x, *params)
        ks_stat, p_value = stats.kstest(data.values, cdf_func)
        return ks_stat, p_value
    except Exception as e:
        print(f"Error performing KS test for {dist_name.upper()}: {e}")
        return None, None


def get_theoretical_lmoments(dist_name: str, fit_params: dict):
    """
    Returns theoretical L-skew (t3) and L-kurtosis (t4) based on the fitted parameters.
    Some distributions may not support nmom=4; we use a conservative map.
    """
    max_nmom_dict = {
        "gev": 4, "exp": 3, "gum": 4, "wei": 4,
        "gpa": 4, "pe3": 3, "gam": 3, "glo": 4
    }
    try:
        dist_class = getattr(distr, dist_name.lower())
        nmom = max_nmom_dict.get(dist_name.lower(), 4)

        theo = dist_class.lmom_ratios(*fit_params.values(), nmom=nmom)

        # lmoments3 may return list/tuple
        if isinstance(theo, (list, tuple)):
            t3 = theo[2] if len(theo) > 2 else None
            t4 = theo[3] if len(theo) > 3 else None
        else:
            t3 = getattr(theo, "t3", None)
            t4 = getattr(theo, "t4", None)

        return t3, t4
    except Exception as e:
        print(f"Error computing theoretical L‐moments for {dist_name.upper()}: {e}")
        return None, None


def select_best_distribution(
    ks_results: dict,
    theoretical_lmoments: dict,
    ensemble_empirical_lmoments: tuple[float | None, float | None],
    threshold_p: float = 0.05
):
    """
    Select best distribution using:
      1) KS test pass across ALL models (p > threshold_p), then
      2) Highest avg KS p-value, then
      3) Smallest avg |t3 - ensemble_t3| and |t4 - ensemble_t4|

    If no distribution passes KS for all models, fallback to the one passing KS for most models.
    """
    selection_details = {}
    candidates = []

    e_t3, e_t4 = ensemble_empirical_lmoments

    for dist_name in ks_results.keys():
        p_values = list(ks_results[dist_name].values())
        if len(p_values) == 0:
            continue

        # Must pass KS for all models to be a "strong" candidate
        if all((pv is not None) and (pv > threshold_p) for pv in p_values):
            avg_p = float(np.mean(p_values))

            t3_diffs, t4_diffs = [], []
            if dist_name in theoretical_lmoments:
                for _, (t3, t4) in theoretical_lmoments[dist_name].items():
                    if (t3 is not None) and (e_t3 is not None):
                        t3_diffs.append(abs(t3 - e_t3))
                    if (t4 is not None) and (e_t4 is not None):
                        t4_diffs.append(abs(t4 - e_t4))

            avg_t3_diff = np.mean(t3_diffs) if len(t3_diffs) else np.nan
            avg_t4_diff = np.mean(t4_diffs) if len(t4_diffs) else np.nan

            # Penalize missing L-moment comparisons
            avg_t3_diff = np.nan_to_num(avg_t3_diff, nan=np.inf)
            avg_t4_diff = np.nan_to_num(avg_t4_diff, nan=np.inf)

            candidates.append({
                "dist_name": dist_name,
                "avg_p_value": avg_p,
                "avg_l_skew_diff": float(avg_t3_diff),
                "avg_l_kurt_diff": float(avg_t4_diff),
            })

    if candidates:
        # Higher p is better, smaller diffs are better
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (x["avg_p_value"], -x["avg_l_skew_diff"], -x["avg_l_kurt_diff"]),
            reverse=True
        )
        best = candidates_sorted[0]["dist_name"]
        selection_details["best_distribution"] = best
        selection_details["criteria"] = candidates_sorted
        return best, selection_details

    # Fallback: choose distribution with most models passing KS
    pass_counts = {
        dist: sum((pv is not None) and (pv > threshold_p) for pv in pvs.values())
        for dist, pvs in ks_results.items()
    }
    best = max(pass_counts, key=pass_counts.get)
    selection_details["best_distribution"] = best
    selection_details["criteria"] = pass_counts
    return best, selection_details


def calculate_quantiles(fit_params: dict, distribution: str, rps: list[int]) -> pd.DataFrame:
    """Return level intensities for given fitted parameters."""
    dist_class = getattr(distr, distribution.lower())
    probs = 1.0 - 1.0 / np.array(rps, dtype=float)
    intensities = dist_class.ppf(probs, *fit_params.values())
    df = pd.DataFrame({"Return Period (Years)": rps, "Quantile": probs, "Intensity": intensities})
    df.set_index("Return Period (Years)", inplace=True)
    return df


def plot_ams_models(annual_max_dict: dict[str, pd.Series], out_png: Path):
    plt.figure(figsize=(12, 6))
    for model, ams in annual_max_dict.items():
        plt.plot(ams.index, ams.values, marker="o", linestyle="-", label=model)
    plt.title(f"Annual Maximum Series (3-day sums) for {scenario.upper()} Models")
    plt.xlabel("Year")
    plt.ylabel("3-day AMS (units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


def plot_ams_obs(obs_ams: pd.Series, out_png: Path):
    plt.figure(figsize=(12, 6))
    plt.plot(obs_ams.index, obs_ams.values, marker="o", linestyle="-", color="purple")
    plt.title(f"Observation 3-day AMS Time Series ({scenario.upper()})")
    plt.xlabel("Year")
    plt.ylabel("3-day AMS (units)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


def plot_ensemble_series(combined_annual_max: pd.DataFrame, out_png: Path):
    ensemble_avg = combined_annual_max.mean(axis=1)
    x_min, x_max = combined_annual_max.index.min(), combined_annual_max.index.max()
    y_min, y_max = combined_annual_max.min().min(), combined_annual_max.max().max()
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    plt.figure(figsize=(17, 8))
    for model in combined_annual_max.columns:
        plt.plot(combined_annual_max.index, combined_annual_max[model], linestyle="-", label=model, alpha=0.5)

    plt.plot(combined_annual_max.index, ensemble_avg, marker="o", linestyle="--", color="black", label="Ensemble Average")

    plt.fill_between(
        combined_annual_max.index,
        combined_annual_max.min(axis=1),
        combined_annual_max.max(axis=1),
        color="gainsboro",
        alpha=0.3,
        label="Model Range"
    )

    plt.title(f"Ensemble of Annual 3-Day Maximum Precipitation Time Series ({scenario.upper()})", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Precipitation (mm)", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


def plot_return_level_curves_with_bootstraps(
    data_for_fitting: np.ndarray,
    return_periods: list[int],
    distribution: str,
    obs_ams: pd.Series | None,
    seed: int,
    n_bootstrap_samples: int,
    title: str,
    alpha_bootstrap: float,
    out_png: Path,
):
    """Return level curve + bootstrapped spread (deterministic via seed)."""
    rng = np.random.default_rng(seed)
    dist_class = getattr(distr, distribution.lower())

    fit_params = dist_class.lmom_fit(data_for_fitting)
    probs = 1.0 - 1.0 / np.array(return_periods, dtype=float)

    all_curves = []
    for _ in range(n_bootstrap_samples):
        bs = rng.choice(data_for_fitting, size=len(data_for_fitting), replace=True)
        fit_res = dist_class.lmom_fit(bs)
        intensities = dist_class.ppf(probs, *fit_res.values())
        all_curves.append(intensities)

    all_curves = np.array(all_curves)
    p5 = np.percentile(all_curves, 5, axis=0)
    median = np.percentile(all_curves, 50, axis=0)
    p95 = np.percentile(all_curves, 95, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    for curve in all_curves:
        ax.plot(return_periods, curve, color="gray", alpha=alpha_bootstrap)

    ax.plot(return_periods, p5, linestyle="--", color="blue", label="5th percentile")
    ax.plot(return_periods, median, linestyle="-", color="green", label="Median")
    ax.plot(return_periods, p95, linestyle="-.", color="red", label="95th percentile")

    if obs_ams is not None and len(obs_ams) > 0:
        obs_fit = dist_class.lmom_fit(obs_ams.values)
        obs_quantiles = dist_class.ppf(probs, *obs_fit.values())
        ax.plot(return_periods, obs_quantiles, linestyle="-", color="purple", label="Observations")

    ax.set_xscale("log")
    ax.set_xlim([min(return_periods), max(return_periods)])
    ax.set_xticks(return_periods)
    ax.set_xticklabels([str(rp) for rp in return_periods])
    ax.set_xlabel("Return Period (Years)", fontsize=13)
    ax.set_ylabel("Precipitation Intensities (mm)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.2, which="both", linestyle="--")
    ax.legend(loc="upper left", fontsize=12, frameon=True, framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_edgecolor("silver")
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


# MAIN PIPELINE
def main():
    ensure_dirs()

    # ---- 1) Load AMS for models ----
    annual_max_dict = {}
    lmom_empirical_dict = {}

    for model_name, file_path in model_files.items():
        ams = load_and_transform_daily_data(file_path)
        if ams is not None and len(ams) > 0:
            annual_max_dict[model_name] = ams
            t3, t4 = compute_empirical_lmoments(ams)
            if (t3 is not None) and (t4 is not None):
                lmom_empirical_dict[model_name] = (t3, t4)

    if not annual_max_dict:
        raise ValueError("No model data loaded — check file paths and formatting.")

    # ---- 2) Align common years ----
    common_years = set.intersection(*[set(ams.index) for ams in annual_max_dict.values()])
    if not common_years:
        raise ValueError("No common years found across models — cannot build ensemble.")
    for model in annual_max_dict:
        annual_max_dict[model] = annual_max_dict[model].loc[sorted(common_years)]
    combined_annual_max = pd.DataFrame(annual_max_dict)

    # ---- 3) Load obs AMS ----
    obs_ams = load_and_transform_daily_data(obs_file)
    if obs_ams is not None and len(obs_ams) > 0:
        obs_t3, obs_t4 = compute_empirical_lmoments(obs_ams)
    else:
        obs_t3 = obs_t4 = None

    # ---- 4) Plots: AMS time series ----
    plot_ams_models(annual_max_dict, FIG_DIR / f"ams_models_{scenario}.png")
    if obs_ams is not None and len(obs_ams) > 0:
        plot_ams_obs(obs_ams, FIG_DIR / f"ams_obs_{scenario}.png")

    # ---- 5) Fit distributions + KS tests ----
    all_fit_parameters = {}
    ks_results = defaultdict(dict)
    theoretical_lmoments_dict = defaultdict(dict)

    for model, data in annual_max_dict.items():
        print(f"\nFitting distributions for model: {model}")
        fit_results = fit_distributions(data, distributions)
        all_fit_parameters[model] = fit_results

        for dist_name, fit in fit_results.items():
            t3, t4 = get_theoretical_lmoments(dist_name, fit)
            if (t3 is not None) or (t4 is not None):
                theoretical_lmoments_dict[dist_name][model] = (t3, t4)

            ks_stat, p_value = perform_ks_test(data, dist_name, fit)
            ks_results[dist_name][model] = p_value if (ks_stat is not None) else 0
            if p_value is not None:
                print(f"{dist_name.upper()}: KS p-value = {p_value:.4f}")
            else:
                print(f"{dist_name.upper()}: KS test failed.")

    # Save KS results
    ks_df = pd.DataFrame(ks_results).T  # dist x model
    ks_df.to_csv(MET_DIR / f"ks_pvalues_{scenario}.csv")
    print(f"\nSaved KS p-values: {MET_DIR / f'ks_pvalues_{scenario}.csv'}")

    # ---- 6) Ensemble empirical L-moments (mean across models) ----
    if lmom_empirical_dict:
        ensemble_t3 = float(np.mean([v[0] for v in lmom_empirical_dict.values()]))
        ensemble_t4 = float(np.mean([v[1] for v in lmom_empirical_dict.values()]))
    else:
        ensemble_t3 = ensemble_t4 = None

    # ---- 7) Auto select best distribution (NO forcing) ----
    best_dist, selection_details = select_best_distribution(
        ks_results=ks_results,
        theoretical_lmoments=theoretical_lmoments_dict,
        ensemble_empirical_lmoments=(ensemble_t3, ensemble_t4),
        threshold_p=0.05,
    )
    print(f"\n{scenario.upper()} Automatic Best Distribution: {best_dist.upper()}")
    pprint(selection_details)

    unified_distribution = best_dist
    print(f"Unified distribution for {scenario.upper()}: {unified_distribution.upper()}")

    # Save selection details (human-readable)
    (MET_DIR / f"selection_details_{scenario}.txt").write_text(str(selection_details), encoding="utf-8")

    # ---- 8) Fit unified distribution to ensemble data (metrics table) ----
    ensemble_data = combined_annual_max.stack().to_numpy()
    dist_class = getattr(distr, unified_distribution.lower())
    fit_ens = dist_class.lmom_fit(ensemble_data)
    ens_return_levels = calculate_quantiles(fit_ens, unified_distribution, return_periods)
    print(f"\n[Ensemble {scenario.upper()}] Fitted '{unified_distribution.upper()}' parameters:")
    pprint(fit_ens)
    print(f"\n[Ensemble {scenario.upper()}] Return Period Intensities (3-day sums):")
    print(ens_return_levels)

    ens_return_levels.to_csv(MET_DIR / f"return_levels_ensemble_{scenario}_{unified_distribution}.csv")

    # ---- 9) Per-model return levels table (your screenshot table) ----
    quantile_results = {}
    prob_levels = 1.0 - 1.0 / np.array(return_periods, dtype=float)

    for model, fit_dict in all_fit_parameters.items():
        if unified_distribution in fit_dict:
            try:
                dist_class = getattr(distr, unified_distribution.lower())
                intensities = dist_class.ppf(prob_levels, *fit_dict[unified_distribution].values())
                quantile_results[model] = intensities
            except Exception as e:
                print(f"Error computing intensities for {model}: {e}")

    quantile_df = pd.DataFrame(quantile_results, index=return_periods).T
    quantile_df.index.name = "Model"
    quantile_df.columns = [f"{rp}-Year" for rp in return_periods]

    print(f"\n{scenario.upper()} Precipitation Intensities using {unified_distribution.upper()} Distribution:")
    print(quantile_df)

    quantile_df.to_csv(MET_DIR / f"return_levels_models_{scenario}_{unified_distribution}.csv")

    # ---- 10) Optional bar chart + heatmap ----
    if PLOT_BAR_CHART:
        sns.set_theme(style="whitegrid")
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(quantile_df.columns)))
        fig, ax = plt.subplots(figsize=(14, 5))
        quantile_df.plot(kind="bar", ax=ax, width=0.8, color=colors, zorder=3)
        ax.set_title(
            f"Precipitation Intensities for Return Periods using {unified_distribution.upper()} ({scenario.upper()})",
            fontsize=16
        )
        ax.set_xlabel("GCMs (Climate Models)", fontsize=15)
        ax.set_ylabel("Precipitation Intensity (mm)", fontsize=15)
        plt.xticks(rotation=0, fontsize=14)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(True, axis="y", alpha=0.2, zorder=1)
        for spine in ax.spines.values():
            spine.set_edgecolor("darkgrey")
            spine.set_linewidth(1.5)
        ax.legend(title="Return Period", fontsize=13, loc="upper left",
                  bbox_to_anchor=(0.01, 0.99), frameon=True, framealpha=0.5)
        plt.tight_layout()
        bar_png = FIG_DIR / f"bar_return_periods_{scenario}_{unified_distribution}.png"
        plt.savefig(bar_png, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved: {bar_png}")

    if PLOT_HEATMAP:
        sns.set_theme(context="notebook", style="white")
        plt.figure(figsize=(8, 5))
        ax = sns.heatmap(quantile_df, cmap="YlOrRd", annot=True, fmt=".1f",
                         cbar_kws={"label": "Intensity (mm)"})
        plt.title(f"Heatmap of Return Period Intensities\n({unified_distribution.upper()} - {scenario.upper()})", fontsize=14)
        plt.xlabel("Return Period", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.tight_layout()
        heat_png = FIG_DIR / f"heatmap_return_periods_{scenario}_{unified_distribution}.png"
        plt.savefig(heat_png, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved: {heat_png}")

    # ---- 11) Ensemble time series plot ----
    plot_ensemble_series(combined_annual_max, FIG_DIR / f"ensemble_ams_{scenario}.png")

    # ---- 12) Return level curves with bootstraps (your screenshot plot) ----
    plot_return_level_curves_with_bootstraps(
        data_for_fitting=ensemble_data,
        return_periods=return_periods,
        distribution=unified_distribution,
        obs_ams=obs_ams if (obs_ams is not None and len(obs_ams) > 0) else None,
        seed=SEED,
        n_bootstrap_samples=1000,
        title=f"Precipitation Intensities using {unified_distribution.upper()} Return Level Curves ({scenario.upper()})",
        alpha_bootstrap=0.01,
        out_png=FIG_DIR / f"return_level_curves_{scenario}_{unified_distribution}.png",
    )


if __name__ == "__main__":
    main()
