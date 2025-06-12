import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=1000):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Standard Cross Entropy
        ce_loss = self.ce(logits, targets)

        # One-hot encode targets
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Apply softmax to logits
        pred = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)

        # Reverse Cross Entropy
        rce_loss = (-1.0 * torch.sum(pred * torch.log(one_hot + 1e-7), dim=1)).mean()

        # SCE Loss = alpha * CE + beta * RCE
        return self.alpha * ce_loss + self.beta * rce_loss

# example usage
# criterion = SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0, num_classes=self.num_classes)
# loss = criterion(logits, y)


def mc_dropout_predict_proba(model, x: torch.Tensor, n_samples: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo dropout: Predict class probabilities and return uncertainty estimates.

    Returns:
        mean_probs (np.ndarray): shape [batch_size, num_classes]
        entropy (np.ndarray): shape [batch_size] — entropy-based uncertainty
        variance (np.ndarray): shape [batch_size] — mean variance across classes (epistemic uncertainty)
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active

    probs = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            probs.append(prob.unsqueeze(0))

    probs = torch.cat(probs, dim=0)             # [n_samples, batch_size, num_classes]
    mean_probs = probs.mean(dim=0)              # [batch_size, num_classes]

    # Entropy: -Σ p log(p)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)  # [batch_size]

    # Variance: mean variance across classes per sample
    variance = probs.var(dim=0).mean(dim=1)      # [batch_size]

    return mean_probs.numpy(), entropy.numpy(), variance.numpy()


def mc_dropout_predict(mean_probs: np.ndarray) -> np.ndarray:
    return np.argmax(mean_probs, axis=1)

# example usage
# probs, entropy_uncertainty, var_uncertainty = mc_dropout_predict_proba(model, test.X, n_samples=25)
# preds = mc_dropout_predict(probs)