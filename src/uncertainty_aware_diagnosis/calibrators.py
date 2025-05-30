import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from pycalib.models.calibrators import LogisticCalibration

# -----------------------------------------------------------------------------
# 1) Base interface
# -----------------------------------------------------------------------------
class BaseCalibrator:
    def fit(self, val_logits: np.ndarray, val_probs: np.ndarray, val_labels: np.ndarray):
        """
        Learn calibration parameters from validation set.

        Args:
            val_logits:  shape (n_val, K), raw MLP logits
            val_probs:   shape (n_val, K), softmax(val_logits)
            val_labels:  shape (n_val,), integer true labels
        """
        raise NotImplementedError

    def predict_proba(self, test_logits: np.ndarray, test_probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to test set.

        Returns:
            - For multiclass: array (n_test, K)
            - For top-label: array (n_test,)
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# 2) Platt's scaling
# -----------------------------------------------------------------------------

class PlattCalibrator(BaseCalibrator):
    def __init__(self, C=0.002, solver="lbfgs", multi_class="multinomial", log_transform=True):
        super().__init__()
        self.cal = LogisticCalibration(
            C=C,
            solver=solver,
            multi_class=multi_class,
            log_transform=log_transform,
        )

    def fit(self, val_logits, val_props, val_labels):
        self.cal.fit(val_props, val_labels)

    def predict_proba(self, test_logits, test_probs):
        return self.cal.predict_proba(test_probs)

# -----------------------------------------------------------------------------
# 3) Multiclass Temperature Scaling
# -----------------------------------------------------------------------------
class MulticlassTemperatureScaling(BaseCalibrator):
    """
    Learns a single temperature T to rescale logits:
      calibrated_logits = logits / T,
      calibrated_probs  = softmax(calibrated_logits)
    by minimizing NLL on the validation set.
    """
    def __init__(self, device=None, init_temp: float = 1.0):
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))
        self.device = device or torch.device("cpu")
        self.temperature.data = self.temperature.data.to(self.device)

    def fit(self, val_logits, val_probs, val_labels):
        # only logits & labels matter
        logits = torch.from_numpy(val_logits).float().to(self.device)
        labels = torch.from_numpy(val_labels).long().to(self.device)

        # ensure T is a parameter on the right device
        self.temperature = nn.Parameter(self.temperature.to(self.device))

        # Optimize T to minimize cross‐entropy
        optimizer = LBFGS([self.temperature], lr=0.1, max_iter=50)
        nll = nn.CrossEntropyLoss()

        def _closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

    def predict_proba(self, test_logits, test_probs):
        logits = torch.from_numpy(test_logits).float().to(self.device)
        with torch.no_grad():
            scaled = logits / self.temperature
            probs  = F.softmax(scaled, dim=1)
        return probs.cpu().numpy()


# -----------------------------------------------------------------------------
# 4) Top-Label (binary) Temperature Scaling
# -----------------------------------------------------------------------------
class TopLabelTemperatureScaling(BaseCalibrator):
    """
    Learns a single temperature T to rescale logits,
    then returns only the maximum softmax confidence: P(correct).
    """
    def __init__(self, device=None, init_temp: float = 1.0):
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))
        self.device = device or torch.device("cpu")
        self.temperature.data = self.temperature.data.to(self.device)

    def fit(self, val_logits, val_probs, val_labels):
        # logits & labels
        logits = torch.from_numpy(val_logits).float().to(self.device)
        labels = torch.from_numpy(val_labels).long().to(self.device)

        # compute “correct” mask once
        with torch.no_grad():
            preds   = torch.argmax(logits, dim=1)
            correct = (preds == labels).float()

        self.temperature = nn.Parameter(self.temperature.to(self.device))

        # Optimize T to minimize binary cross‐entropy on top-label confidence
        optimizer = LBFGS([self.temperature], lr=0.1, max_iter=50)
        bce = nn.BCELoss()

        def _closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            probs  = F.softmax(scaled, dim=1)
            top_p, _ = probs.max(dim=1)
            loss = bce(top_p, correct)
            loss.backward()
            return loss

        optimizer.step(_closure)

    def predict_proba(self, test_logits, test_probs):
        logits = torch.from_numpy(test_logits).float().to(self.device)
        with torch.no_grad():
            scaled = logits / self.temperature
            probs  = F.softmax(scaled, dim=1)
            top_p, _ = probs.max(dim=1)
        return top_p.cpu().numpy()