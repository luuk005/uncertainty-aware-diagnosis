import numpy as np  # type: ignore
import torch  # type: ignore
from typing import List, Dict, Union, Tuple  # type: ignore
from torchmetrics import F1Score, Recall  # type: ignore
from sklearn.metrics import brier_score_loss  # type: ignore
from pycalib.metrics import classwise_ECE, conf_ECE  # type: ignore
from pycalib.visualisations import plot_reliability_diagram  # type: ignore


class Evaluator:
    """
    Evaluate classification and calibration performance using metrics such as:
    - F1 Macro
    - Recall Macro
    - Brier Score Loss
    - Expected Calibration Error (ECE)
    - Reliability Diagrams
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def compute_classification_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """Compute F1 and Recall macro scores."""
        f1 = F1Score(task="multiclass", average="macro", num_classes=self.num_classes)
        recall = Recall(task="multiclass", average="macro", num_classes=self.num_classes)
        f1.update(y_pred, y_true)
        recall.update(y_pred, y_true)
        return {
            "F1 Macro": f1.compute().item(),
            "Recall Macro": recall.compute().item()
        }

    def compute_brier_score(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """Compute Brier Score for binary decisions."""
        binary_true = (np.argmax(probs, axis=1) == y_true).astype(int)
        return brier_score_loss(binary_true, probs.max(axis=1))

    def compute_ece_metrics(
        self,
        y_true: np.ndarray,
        prob_list: List[np.ndarray],
        metric_names: List[str],
        bins: int = 15
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple ECE metrics for a list of probability outputs.

        Parameters:
        - y_true: True labels
        - prob_list: List of probability arrays from different models
        - metric_names: Names for those models (used in keys)
        - bins: Number of bins for ECE
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, probs in zip(metric_names, prob_list):
            results[name] = {
                "conf_ECE": conf_ECE(y_true, probs, bins=bins),
                "classwise_ECE": classwise_ECE(y_true, probs, bins=bins)
            }
        return results

    def plot_reliability(
        self,
        y_true: np.ndarray,
        probs_list: List[np.ndarray],
        legend: List[str],
        bins: int = 15
    ) -> None:
        """Plot reliability diagram with confidence and histogram."""
        plot_reliability_diagram(
            labels=y_true,
            scores=probs_list,
            legend=legend,
            confidence=True,
            bins=bins
        )

    def evaluate_all(
        self,
        y_true: Union[torch.Tensor, np.ndarray],
        prob_dict: Dict[str, np.ndarray],
        bins: int = 15
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Full evaluation pipeline for multiple models.

        Parameters:
        - y_true: Ground-truth labels (PyTorch or NumPy)
        - prob_dict: Dictionary of {model_name: prob_array}
        - bins: Bins for calibration metrics

        Returns:
        - Tuple of two dictionaries: classification results and ECE scores
        """
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.numpy()
        else:
            y_true_np = y_true

        results: Dict[str, Dict[str, float]] = {}
        for model_name, probs in prob_dict.items():
            y_pred = torch.argmax(torch.tensor(probs), dim=1)
            class_metrics = self.compute_classification_metrics(torch.tensor(y_true_np), y_pred)
            brier = self.compute_brier_score(y_true_np, probs)
            results[model_name] = {
                **class_metrics,
                "Brier Score Loss": brier
            }

        print("\n====== Classification Metrics ======\n")
        print(f"{'Model':<15} {'F1 Macro':>12} {'Recall':>12} {'Brier':>12}")
        print("-" * 45)
        for model, metrics in results.items():
            print(f"{model:<15} {metrics['F1 Macro']:>12.4f}    {metrics['Recall Macro']:>12.4f}    {metrics['Brier Score Loss']:>12.4f}")

        print("\n====== Calibration (Expected Calibration Error) Metrics ======\n")
        ece_scores = self.compute_ece_metrics(y_true_np, list(prob_dict.values()), list(prob_dict.keys()), bins=bins)
        print(f"{'Model':<15} {'conf_ECE':>12} {'classwise_ECE':>16}")
        print("-" * 45)
        for model, scores in ece_scores.items():
            print(f"{model:<15} {scores['conf_ECE']:>12.4f}    {scores['classwise_ECE']:>16.4f}")

        print("\n====== Reliability Diagram ======\n")
        self.plot_reliability(y_true_np, list(prob_dict.values()), list(prob_dict.keys()), bins=bins)

        return results, ece_scores
