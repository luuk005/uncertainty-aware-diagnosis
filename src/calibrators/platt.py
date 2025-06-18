from calibrators.base import BaseCalibrator  # type: ignore

import numpy as np  # type: ignore
from pycalib.models.calibrators import LogisticCalibration  # type: ignore


class PlattCalibrator(BaseCalibrator):
    """
    Implements Platt scaling using a logistic regression model for calibration.
    PlattCalibrator (wrapper for LogisticCalibration)
    """

    def __init__(
      self,
      C: float = 0.03,
      solver: str = "lbfgs",
      log_transform: bool = True,
    ):
        super().__init__()
        self.C = C
        self.solver = solver
        self.log_transform = log_transform
        self.cal = LogisticCalibration(
            C=C,
            solver=solver,
            log_transform=log_transform,
        )

    def fit(
      self,
      val_props: np.ndarray,
      val_labels: np.ndarray
    ) -> "PlattCalibrator":  # no test_logits
        self.cal.fit(val_props, val_labels)
        return self

    def predict_proba(
      self,
      test_probs: np.ndarray
    ) -> np.ndarray:  # no test_logits
        return self.cal.predict_proba(test_probs)
