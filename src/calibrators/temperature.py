from calibrators.base import BaseCalibrator  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.optim import LBFGS  # type: ignore


class TemperatureScaling(BaseCalibrator):
    """
    Temperature scaling calibrator using PyTorch.
    """

    def __init__(
        self,
        init_temp: float = 1.5,
        lr: float = 0.1,
        max_iter: int = 50,
        device=None,
    ):
        super().__init__()
        self.init_temp = init_temp
        self.lr = lr
        self.max_iter = max_iter
        self.device = device or torch.device("cpu")
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
    ) -> "TemperatureScaling":
        logits = torch.from_numpy(val_logits).float().to(self.device)
        labels = torch.from_numpy(val_labels).long().to(self.device)

        self.temperature = nn.Parameter(self.temperature.to(self.device))
        optimizer = LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = loss_fn(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_proba(
        self,
        test_logits: np.ndarray,
    ) -> np.ndarray:
        logits_tensor = torch.from_numpy(test_logits).float().to(self.device)
        scaled = logits_tensor / self.temperature
        return F.softmax(scaled, dim=1).detach().cpu().numpy()


class TopLabelTemperature(BaseCalibrator):
    """
    Modular Temperature Scaling: supports both standard and top-label calibration.
    Top-label calibration adjusts the predicted probabilities of a classifier
    by focusing on the logits of the top predicted class, improving accuracy for the true label.
    """

    def __init__(
        self,
        mode: str = "top_label",  # or "standard"
        device=None,
        init_temp: float = 1.0,
        lr: float = 0.1,
        max_iter: int = 50,
    ):
        super().__init__()
        assert mode in {"standard", "top_label"}, "mode must be 'standard' or 'top_label'"
        self.mode = mode
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))
        self.device = device or torch.device("cpu")
        self.temperature.data.fill_(1.5)
        self.lr = lr
        self.max_iter = max_iter

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
    ) -> "TopLabelTemperature":
        logits = torch.from_numpy(val_logits).float().to(self.device)
        labels = torch.from_numpy(val_labels).long().to(self.device)

        self.temperature = nn.Parameter(self.temperature.to(self.device))
        optimizer = LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        if self.mode == "standard":
            loss_fn = nn.CrossEntropyLoss()

            def _eval():
                optimizer.zero_grad()
                scaled = logits / self.temperature
                loss = loss_fn(scaled, labels)
                loss.backward()
                return loss

        elif self.mode == "top_label":
            pred_class = torch.argmax(logits, dim=1)
            correct = (pred_class == labels).float()
            top_logits = torch.gather(logits, 1, pred_class.unsqueeze(1)).squeeze(1)
            loss_fn = nn.BCEWithLogitsLoss()

            def _eval():
                optimizer.zero_grad()
                scaled = top_logits / self.temperature
                loss = loss_fn(scaled, correct)
                loss.backward()
                return loss

        optimizer.step(_eval)

    def predict_proba(
        self,
        test_logits: np.ndarray,
    ) -> np.ndarray:
        logits_tensor = torch.from_numpy(test_logits).float().to(self.device)
        scaled = logits_tensor / self.temperature
        return F.softmax(scaled, dim=1).detach().cpu().numpy()

    def predict_top_confidence(
        self,
        test_logits: np.ndarray,
    ) -> np.ndarray:
        logits_tensor = torch.from_numpy(test_logits).float().to(self.device)
        scaled = logits_tensor / self.temperature
        probs = F.softmax(scaled, dim=1)
        return probs.max(dim=1).values.detach().cpu().numpy()
