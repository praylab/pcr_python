import matplotlib.pyplot as plt
import numpy as np

from pcr.model import PCRModel


def plot_exceedance(
    model: PCRModel,
    years: list = [25, 50, 75, 100],
    colors: list = ['k', 'k', 'r', 'r'],
    dashes: list = ['-', '--', '-', '--'],
    ax: plt.Axes = None,
) -> plt.Axes:
    '''
    plot recession exceedance probability curves for a set of horizon years

    for each year in `years`, sorts the simulated shoreline recession
    (model.shoreline_stats) descending and plots it against exceedance
    probability (0-100 %) on a log y-axis.
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    exceedance = np.linspace(0, 100, model.nr_simulation)

    for year, color, dash in zip(years, colors, dashes):
        sorted_year = np.sort(-model.shoreline_stats[year - 1])[::-1]
        ax.plot(sorted_year, exceedance, label=str(year + model.year_start), color=color, linestyle=dash)

    ax.set_yscale('log')
    ax.set_ylim([1, 100])
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylabel('Exceendance Probability (%)')
    ax.set_xlabel('Recession (m)')
    ax.legend()

    return ax
