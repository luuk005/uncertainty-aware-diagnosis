from __future__ import annotations   # type: ignore

import torch   # type: ignore
from torch.utils.data import Sampler   # type: ignore

from pathlib import Path   # type: ignore
from typing import Sequence, Optional, Union   # type: ignore

import plotly.graph_objects as go   # type: ignore
import pandas as pd   # type: ignore


class CurriculumSampler(Sampler):
    """Iterates samples whose difficulty <= tau, in ascending difficulty order"""
    def __init__(self, difficulties: torch.Tensor, tau: float):
        self.diff = difficulties
        self.tau = tau

    def set_tau(self, tau: float):
        self.tau = tau

    def __iter__(self):
        idxs = torch.arange(len(self.diff))
        allow = idxs[self.diff <= self.tau]
        sorted_allow = allow[torch.argsort(self.diff[allow])]  # easy -> hard
        return iter(sorted_allow.tolist())

    def __len__(self):
        return int((self.diff <= self.tau).sum())


def plot_curriculum_schedule(
    *,
    f1_scores: Sequence[float],
    tau_values: Sequence[float],
    epochs: Optional[Sequence[int]] = None,
    title: str = "Curriculum Learning: Tau Threshold vs Validation F1",
    f1_label: str = "Validation F1",
    tau_label: str = "Tau (Difficulty Threshold)",
    xlabel: str = "Epoch",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    height: int = 500,
    width: int = 900,
) -> go.Figure:
    """
    Interactive Plotly version of curriculum-learning progress plot.
    Shows validation‐set F1 score vs. the evolving τ threshold.

    Parameters
    ----------
    f1_scores:
        Sequence of validation F1 scores recorded after each epoch.
    tau_values:
        Sequence of τ thresholds used in the same epochs.
    epochs:
        Optional explicit x-axis values. If None, uses range(len(f1_scores)).
    title, f1_label, tau_label, xlabel:
        Text for the chart title and axis labels.
    save_path:
        If provided, the figure is saved as an HTML or image file.
    show:
        Whether to call `fig.show()` (default: True).
    height, width:
        Size of the Plotly figure in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if len(f1_scores) != len(tau_values):
        raise ValueError("f1_scores and tau_values must be the same length.")

    if epochs is None:
        epochs = list(range(len(f1_scores)))
    elif len(epochs) != len(f1_scores):
        raise ValueError("epochs must be the same length as f1_scores.")

    df = pd.DataFrame({
        "Epoch": epochs,
        "F1": f1_scores,
        "Tau": tau_values,
    })

    fig = go.Figure()

    # F1 Score trace
    fig.add_trace(go.Scatter(
        x=df["Epoch"],
        y=df["F1"],
        mode="lines+markers",
        name=f1_label,
        yaxis="y1",
        marker=dict(symbol="circle", size=8),
        line=dict(color="blue"),
    ))

    # Tau threshold trace
    fig.add_trace(go.Scatter(
        x=df["Epoch"],
        y=df["Tau"],
        mode="lines+markers",
        name=tau_label,
        yaxis="y2",
        marker=dict(symbol="square", size=8),
        line=dict(color="red", dash="dash"),
    ))

    # Layout with dual y-axes
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family="Arial")),
        xaxis=dict(title=xlabel),
        yaxis=dict(title=f1_label, side="left", color="blue"),
        yaxis2=dict(
            title=tau_label,
            overlaying="y",
            side="right",
            color="red"
        ),
        height=height,
        width=width,
        legend=dict(x=0.01, y=1.14),
        margin=dict(l=60, r=60, t=70, b=50),
    )

    # Save (HTML or image)
    if save_path:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ext = save_path.suffix.lower()
        if ext == ".html":
            fig.write_html(str(save_path))
        elif ext in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            fig.write_image(str(save_path), scale=2)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    if show:
        fig.show()

    return fig





