from calibrators.base import BaseCalibrator  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.optim import LBFGS  # type: ignore


class DirichletCalibration(BaseCalibrator):
    """
    Dirichlet calibration using linear transformation and softmax.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float = 0.01,
        max_iter: int = 100,
        device=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.max_iter = max_iter
        self.device = device or torch.device("cpu")
        self.A = nn.Parameter(torch.eye(num_classes, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))
        self._initialized = False

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
    ) -> "DirichletCalibration":
        logits = torch.tensor(val_logits, dtype=torch.float32, device=self.device)
        labels = torch.tensor(val_labels, dtype=torch.long, device=self.device)

        self.A = nn.Parameter(self.A.to(self.device))
        self.b = nn.Parameter(self.b.to(self.device))

        optimizer = LBFGS([self.A, self.b], lr=self.lr, max_iter=self.max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            transformed = logits @ self.A.T + self.b
            loss = loss_fn(transformed, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._initialized = True
        return self

    def predict_proba(
        self,
        test_logits: np.ndarray,
    ) -> np.ndarray:
        assert self._initialized, "fit() must be called before predict_proba()"
        logits_tensor = torch.tensor(test_logits, dtype=torch.float32, device=self.device)
        transformed = logits_tensor @ self.A.T + self.b
        probs = F.softmax(transformed, dim=1)
        return probs.detach().cpu().numpy()
