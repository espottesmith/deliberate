from collections import Counter
from typing import List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns


def plot_bar(
    data: List,
    x_label: str,
    y_label: str = "Count",
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    font_scale: float = 1.3,
    fig_size: Tuple[float, float] = (8.0, 6.0),
    filename: Optional[Path] = None,
    num_bins: Optional[int] = None,
    **kwargs
) -> Axes:
    """
    Args:
        data (List): Data to be plotted
        x_label (str):
        y_label (str): Default is "Count"
        x_shift (float): How much should count labels be shifted in horizontal direction. Default is 0.
        y_shift (float): How much should count labels be shifted in vertical direction. Default is 0.
        font_scale (float): Scaling factor for text. Default is 1.3
        fig_size (Tuple[float, float]): Size of figure in inches. Default is (8.0, 6.0)
        filename (Optional[Path]): If not None (default), the figure will be output to this file path.
        num_bins (Optional[int]): For a histogram where data are grouped into binds, this determines how
            many bins should be made.
    
    Returns:
        ax (matplotlib.axes.Axes)
    
    
    """
    sns.set(
        context="notebook",
        style="darkgrid",
        font_scale=font_scale,
        rc={"figure.figsize": fig_size},
    )

    counter = dict(Counter(data))
    keys = sorted(counter.keys())
    y = [counter[k] for k in keys]

    if num_bins is None:
        ax = sns.barplot(x=keys, y=y, **kwargs)
    else:
        ax = sns.histplot(data, stat="count", bins=num_bins, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if num_bins is None:
        # annotate the bar plot with the count
        for i, k in enumerate(keys):
            ax.annotate(counter[k], (i + x_shift, counter[k] + y_shift))

        # adjust y range to give more space for the annotation
        ax.set_ylim(0, max(y) * 1.1)

    # save to file
    if filename is not None:
        figdir = Path("figures").resolve()
        if not figdir.exists():
            figdir.mkdir()
        filename = figdir.joinpath(filename)

        fig = ax.get_figure()
        fig.savefig(filename.as_posix(), bbox_inches="tight")

    return ax