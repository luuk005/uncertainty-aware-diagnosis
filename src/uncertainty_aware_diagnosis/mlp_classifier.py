import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.nn.init as init  # type: ignore
from torch.utils.data import DataLoader, Subset  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from torchmetrics import F1Score, Recall  # type: ignore
import numpy as np  # type: ignore
from torch.optim import LBFGS
from pycalib.models.calibrators import LogisticCalibration  # type: ignore
from .noise_robustness_improvements import SymmetricCrossEntropyLoss


class SimpleMLP(nn.Module):
    """A simple multi-layer perceptron (MLP) for classification tasks.

    This model consists of one hidden layer with dropout regularization and an output layer
    for class predictions. The hidden layer uses the ELU activation function.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of neurons in the hidden layer.
        num_classes (int): The number of output classes for classification.
        dropout (float, optional): The dropout probability. Default is 0.2.
        device (str, optional): The device to run the model on. Default is 'cpu'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.classes_ = None
        self.hidden = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.output = nn.Linear(hidden_dim, num_classes).to(self.device)
        init.kaiming_normal_(self.hidden.weight)
        init.kaiming_normal_(self.output.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing the logits.
        """
        h = F.elu(self.hidden(x))
        h = self.dropout(h)
        logits = self.output(h)
        return logits

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        early_stopping_patience: int | None = None,
        verbose: bool = True,
    ):
        """Train the SimpleMLP.

        Args:
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            num_epochs (int): Number of epochs to train the model.
            learning_rate (float): Learning rate for the optimizer.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        # Define the loss function and optimizer
        # criterion = nn.CrossEntropyLoss()  # Use appropriate loss function
        criterion = SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0, num_classes=self.num_classes) # noise robustness improvement
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Metrics
        f1_macro = F1Score(
            task="multiclass", average="macro", num_classes=self.output.out_features
        )
        recall_macro = Recall(
            task="multiclass", average="macro", num_classes=self.output.out_features
        )

        # Early stopping variables
        best_f1 = -1
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            # Training loop
            self.train()
            running_loss = 0.0
            for batch_idx, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                logits = self(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)
            avg_train_loss = running_loss / len(train_loader.dataset)

            # Validation loop
            self.eval()
            val_loss = 0.0
            f1_macro.reset()
            recall_macro.reset()
            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(val_loader):
                    logits = self(X)
                    val_loss += criterion(logits, y).item() * X.size(0)
                    y_pred = torch.argmax(logits, dim=1)
                    f1_macro.update(y_pred, y)
                    recall_macro.update(y_pred, y)
            avg_val_loss = val_loss / len(val_loader.dataset)

            if verbose:
                # Print progress
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"F1 Macro: {f1_macro.compute():.4f}, Recall Macro: {recall_macro.compute():.4f}"
                )

            # Check for improvement for early stopping
            if early_stopping_patience is not None:
                if f1_macro.compute() > best_f1:
                    best_f1 = f1_macro.compute()
                    best_model_state = self.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)  # Load best model state

        # Set classes_ attribute
        self.classes_ = torch.unique(torch.cat([y for _, y in train_loader])).numpy()

        if verbose:
            print(
                "=================================================================================\n"
            )
            print(
                f"Best F1 Macro: {f1_macro.compute():.4f}, Recall Macro: {recall_macro.compute():.4f}"
            )

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict the class for the input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,) containing the predicted class indices.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self(x)  # Forward pass
            predicted_classes = torch.argmax(
                logits, dim=1
            )  # Get the class with the highest probability
        return predicted_classes.numpy()

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict class probabilities for the input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing the probabilities.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self(x)  # Forward pass
            probabilities = F.softmax(
                logits, dim=1
            )  # Apply softmax to get probabilities
        return probabilities.numpy()
    
    def predict_logits(self, x: torch.Tensor) -> np.ndarray:
        """Return raw logits (pre-softmax scores) for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            np.ndarray: Raw logits of shape (batch_size, num_classes).
        """
        self.eval()  # Ensure the model is in eval mode
        with torch.no_grad():
            logits = self(x.to(self.device))
        return logits.detach().cpu().numpy()


    def fit_cv(
        self,
        dataset: torch.utils.data.Dataset,
        k_folds: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        early_stopping_patience: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """
        Perform k-fold cross-validation on the entire dataset and report per-fold metrics.

        Args:
            dataset: a torch.utils.data.Dataset yielding (features, label) tuples.
            k_folds (int): number of folds (e.g., 3 for 3-fold CV).
            batch_size (int): batch size for DataLoader.
            num_epochs (int): epochs to train each fold.
            learning_rate (float): learning rate for optimizer.
            early_stopping_patience (int, optional): early stopping on validation F1 Macro.
            verbose (bool): whether to print progress per epoch and per fold.

        Returns:
            A list of length k_folds, each entry is a dict:
                {"f1_macro": float, "recall_macro": float}
            and prints the average across all folds.
        """
        # Prepare K-Fold splitter
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results: list[dict[str, float]] = []

        # Enumerate folds
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            if verbose:
                print(f"\n=== Fold {fold_idx + 1}/{k_folds} ===")

            # Subset the original dataset for this fold
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)

            # DataLoaders for this fold
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # Instantiate a fresh model for this fold
            model = SimpleMLP(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
                dropout=self.dropout_rate,
                device=str(self.device),
            )
            model.to(self.device)

            # Train on this fold
            model.fit(
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                verbose=verbose,
            )

            # After training, compute metrics on the validation split
            model.eval()
            f1_metric = F1Score(
                task="multiclass", average="macro", num_classes=self.num_classes
            ).to(self.device)
            recall_metric = Recall(
                task="multiclass", average="macro", num_classes=self.num_classes
            ).to(self.device)

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = model(X_batch)
                    preds = torch.argmax(logits, dim=1)
                    f1_metric.update(preds, y_batch)
                    recall_metric.update(preds, y_batch)

            fold_f1 = f1_metric.compute().cpu().item()
            fold_recall = recall_metric.compute().cpu().item()

            if verbose:
                print(
                    f"Fold {fold_idx + 1} → F1 Macro: {fold_f1:.4f} | Recall Macro: {fold_recall:.4f}"
                )

            fold_results.append({"f1_macro": fold_f1, "recall_macro": fold_recall})

        # Compute average across folds
        avg_f1 = sum(r["f1_macro"] for r in fold_results) / k_folds
        avg_recall = sum(r["recall_macro"] for r in fold_results) / k_folds
        if verbose:
            print(f"\n=== Average across {k_folds} folds ===")
            print(
                f"Average F1 Macro: {avg_f1:.4f} | Average Recall Macro: {avg_recall:.4f}\n"
            )

        return fold_results
    
    def mc_predict_proba(self, x: torch.Tensor, n_samples: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout: Predict class probabilities and return uncertainty estimates.

        Returns:
            mean_probs (np.ndarray): shape [batch_size, num_classes]
            entropy (np.ndarray): shape [batch_size] — entropy-based uncertainty
            variance (np.ndarray): shape [batch_size] — mean variance across classes (epistemic uncertainty)
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active

        probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(x)
                prob = F.softmax(logits, dim=1)
                probs.append(prob.unsqueeze(0))

        probs = torch.cat(probs, dim=0)             # [n_samples, batch_size, num_classes]
        mean_probs = probs.mean(dim=0)              # [batch_size, num_classes]

        # Entropy: -Σ p log(p)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)  # [batch_size]

        # Variance: mean variance across classes per sample
        variance = probs.var(dim=0).mean(dim=1)      # [batch_size]

        return mean_probs.numpy(), entropy.numpy(), variance.numpy()
    
    def mc_predict(self, x: torch.Tensor, n_samples: int = 25) -> np.ndarray:
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active

        probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(x)
                prob = F.softmax(logits, dim=1)
                probs.append(prob.unsqueeze(0))

        probs = torch.cat(probs, dim=0)             # [n_samples, batch_size, num_classes]
        mean_probs = probs.mean(dim=0)

        return np.argmax(mean_probs, axis=1)
    

    def mc_predict_logits(self, x: torch.Tensor, n_samples: int = 25) -> tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout: Predict logits with uncertainty estimate (variance over logits).

        Returns:
            mean_logits (np.ndarray): shape [batch_size, num_classes]
            logit_variance (np.ndarray): shape [batch_size] — mean variance across logits per sample
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        logits_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(x)  # shape [batch_size, num_classes]
                logits_list.append(logits.unsqueeze(0))  # shape [1, batch_size, num_classes]

        logits_mc = torch.cat(logits_list, dim=0)     # shape [n_samples, batch_size, num_classes]
        mean_logits = logits_mc.mean(dim=0)           # shape [batch_size, num_classes]
        logit_var = logits_mc.var(dim=0).mean(dim=1)  # shape [batch_size]

        return mean_logits.numpy(), logit_var.numpy()




class PlattCalibrator:
    """
    Platt's scaling calibrator, which is a technique used to calibrate the output probabilities of a classifier.
    It involves a single scalar parameter that uniformly scales all logits, which can be optimized
    to improve calibration.
    """
    def __init__(
        self, C=0.03, solver="lbfgs", log_transform=True
    ):
        super().__init__()
        self.cal = LogisticCalibration(
            C=C,
            solver=solver,
            # multi_class=multi_class,
            log_transform=log_transform,
        )

    def fit(self, val_props, val_labels): # no val_logits
        self.cal.fit(val_props, val_labels)

    def predict_proba(self, test_probs): # no test_logits
        return self.cal.predict_proba(test_probs)


class TemperatureScaling:
    """
    Temperature-scaling calibrator, which is a technique used to calibrate the output probabilities of a classifier.
    It involves a single scalar parameter that uniformly scales all logits, which can be optimized
    to improve calibration.
    """

    def __init__(
        self,
        device=None,
        init_temp: float = 1.0,
        lr: float = 0.1,
        max_iter: int = 50,
    ):
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))
        self.device = device or torch.device("cpu")
        self.temperature.data.fill_(1.5)  # good starting point
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, logits_val: np.ndarray, labels_val: np.ndarray):
        """
        logits_val: (n_val, n_classes) raw network outputs
        labels_val: (n_val,) integer labels
        """
        logits = torch.from_numpy(logits_val).float().to(self.device)
        labels = torch.from_numpy(labels_val).long().to(self.device)
        self.temperature = nn.Parameter(self.temperature.to(self.device))

        # use LBFGS to optimize T
        optimizer = LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        nll_criterion = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        # no need to return; self.temperature is updated in‐place

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits_tensor = torch.from_numpy(logits).float().to(self.device)
        scaled = logits_tensor / self.temperature
        return F.softmax(scaled, dim=1).detach().cpu().numpy()


class TopLabelTemperature:
    """
    Modular Temperature Scaling: supports both standard and top-label calibration.
    """

    def __init__(
        self,
        mode: str = "top_label",  # or "standard"
        device=None,
        init_temp: float = 1.0,
        lr: float = 0.1,
        max_iter: int = 50,
    ):
        assert mode in {"standard", "top_label"}, "mode must be 'standard' or 'top_label'"
        self.mode = mode
        self.temperature = nn.Parameter(torch.tensor([init_temp], dtype=torch.float32))
        self.device = device or torch.device("cpu")
        self.temperature.data.fill_(1.5)
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, logits_val: np.ndarray, labels_val: np.ndarray):
        logits = torch.from_numpy(logits_val).float().to(self.device)
        labels = torch.from_numpy(labels_val).long().to(self.device)
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

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits_tensor = torch.from_numpy(logits).float().to(self.device)
        scaled = logits_tensor / self.temperature
        return F.softmax(scaled, dim=1).detach().cpu().numpy()

    def predict_top_confidence(self, logits: np.ndarray) -> np.ndarray:
        logits_tensor = torch.from_numpy(logits).float().to(self.device)
        scaled = logits_tensor / self.temperature
        probs = F.softmax(scaled, dim=1)
        return probs.max(dim=1).values.detach().cpu().numpy()
    

class DirichletCalibration:
    def __init__(self, num_classes, device=None, lr=0.01, max_iter=100):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.lr = lr
        self.max_iter = max_iter

        self.A = nn.Parameter(torch.eye(num_classes, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))
        self._initialized = False

    def fit(self, logits_val: np.ndarray, labels_val: np.ndarray):
        """
        Fit the Dirichlet calibration model on validation logits and true labels.
        Args:
            logits_val: (n_samples, num_classes), raw logits
            labels_val: (n_samples,), integer class labels
        """
        logits = torch.tensor(logits_val, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels_val, dtype=torch.long, device=self.device)

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

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Calibrate logits using the learned Dirichlet parameters and return probabilities.
        Args:
            logits: (n_samples, num_classes)
        Returns:
            np.ndarray: calibrated probabilities (n_samples, num_classes)
        """
        assert self._initialized, "Call fit() before predict_proba()"
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=self.device)
        transformed = logits_tensor @ self.A.T + self.b
        probs = F.softmax(transformed, dim=1)
        return probs.detach().cpu().numpy()