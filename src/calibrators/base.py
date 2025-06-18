from abc import ABC, abstractmethod  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
import numpy as np  # type: ignore


class BaseCalibrator(ABC, BaseEstimator):
    """
    Abstract base class for all calibrators. Implements the sklearn interface.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseCalibrator":
        """
        Fit the calibrator to the calibration/validation data.

        Parameters:
        ----------
        X : np.ndarray
            The input predictions or logits.
        y : np.ndarray
            The true labels.

        Returns:
        -------
        self : BaseCalibrator
            Fitted calibrator instance.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
         Predict the calibrated probabilities for the inference/test data.

        Parameters:
        ----------
        X : np.ndarray
            Input predictions or logits.

        Returns:
        -------
        np.ndarray
            Calibrated probabilities.
        """
        pass
