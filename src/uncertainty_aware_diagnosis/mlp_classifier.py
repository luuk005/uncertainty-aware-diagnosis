import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import F1Score, Recall
from typing import Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class sklearnMLP(BaseEstimator, ClassifierMixin, nn.Module):
    _estimator_type = "classifier"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        early_stopping_patience: int | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.verbose = verbose

        # define layers
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.hidden.weight)
        nn.init.kaiming_normal_(self.output.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.hidden(x))
        h = self.dropout(h)
        return self.output(h)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # convert arrays → tensors/dataloader
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.int64))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # move to device
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        best_f1 = -np.inf
        epochs_no_improve = 0
        best_state = None

        for epoch in range(self.num_epochs):
            # train
            self.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(loader.dataset)

            # optionally early‐stop on training set F1 as a proxy
            if self.early_stopping_patience is not None:
                # compute F1 on train (or you could split out a val set)
                from torchmetrics import F1Score
                self.eval()
                f1 = F1Score(task="multiclass", average="macro", num_classes=self.num_classes).to(self.device)
                with torch.no_grad():
                    for xb, yb in loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        preds = torch.argmax(self(xb), dim=1)
                        f1.update(preds, yb)
                current_f1 = f1.compute().item()
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_state = self.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Loss={avg_loss:.4f}, F1={current_f1:.4f}")
            elif self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss={avg_loss:.4f}")

        # restore best
        if best_state is not None:
            self.load_state_dict(best_state)

        # required by sklearn
        self.classes_ = np.unique(y.numpy())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self(X_t)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()


class SklearnWrapper(BaseEstimator, ClassifierMixin):
    """sklearn‐style wrapper around your PyTorch SimpleMLP."""
    _estimator_type = "classifier"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.verbose = verbose

        # instantiate your PyTorch model here
        self._torch_model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # → 1) build a DataLoader from the arrays
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # → 2) call your existing PyTorch .fit(...) loop
        #     here we just pass the same loader as “validation” too,
        #     but you can split off a hold-out if you like.
        self._torch_model.fit(
            train_loader=loader,
            val_loader=loader,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )

        # → 3) expose what sklearn needs
        self.classes_ = self._torch_model.classes_
        self.n_features_in_ = X.shape[1]
        return self

    # def predict(self, X: np.ndarray) -> np.ndarray:
    #     # convert, do a forward‐pass & argmax
    #     X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
    #     preds = self._torch_model.predict(X_t)
    #     return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # convert, forward‐pass + softmax
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        probs = self._torch_model.predict_proba(X_t)
        return probs
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        #Compute the decision function (logits) for the input data.
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._torch_model(X_t)  # Get logits from the model
        return logits.cpu().numpy()  # Return as numpy array


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
            device: str = 'cpu',
    ):
        super(SimpleMLP, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
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

    def fit(self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int, 
            learning_rate: float, 
            early_stopping_patience: int | None = None, 
            verbose: bool = True
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
        criterion = nn.CrossEntropyLoss()  # Use appropriate loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Metrics
        f1_macro = F1Score(task="multiclass", average='macro', num_classes=self.output.out_features)
        recall_macro = Recall(task="multiclass", average='macro', num_classes=self.output.out_features)

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
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                    f'F1 Macro: {f1_macro.compute():.4f}, Recall Macro: {recall_macro.compute():.4f}')

            # Check for improvement for early stopping
            if early_stopping_patience is not None:
                if f1_macro.compute() > best_f1:
                    best_f1 = f1_macro.compute()
                    best_model_state = self.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        print(f'Early stopping triggered at epoch {epoch + 1}')
                        break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)  # Load best model state

        # Set classes_ attribute
        self.classes_ = torch.unique(torch.cat([y for _, y in train_loader])).numpy()
  
        if verbose:
            print('=================================================================================\n')
            print(f'Best F1 Macro: {f1_macro.compute():.4f}, Recall Macro: {recall_macro.compute():.4f}')

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
            predicted_classes = torch.argmax(logits, dim=1)  # Get the class with the highest probability
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
            probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        return probabilities.numpy()

class MLPClassifier(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int, 
            num_classes: int, 
            p: float = 0.2,
            use_embedding: bool = False,
            num_codes: Optional[int] = None,
            embed_dim: Optional[int] = None
    ):
        """
        input_dim: Dimensionality of input features (excluding procedure_code if using embedding).
        hidden_dim: Number of neurons in the hidden layer.
        num_classes: Number of output classes for classification.
        use_embedding: If True, use an nn.Embedding for the procedure_code feature.
        high_card_codes: Number of unique procedure codes (required if use_embedding=True).
        embed_dim: Dimension of the embedding vector (required if use_embedding=True).
        """
        super(MLPClassifier, self).__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            assert num_codes is not None and embed_dim is not None, \
                "high_card_codes and embed_dim must be provided when use_embedding is True"
            # Embedding layer for high-cardinality categorical feature
            self.hc1_embedding = nn.Embedding(num_codes, embed_dim) # only use first high cardinal category
            # Adjust input dimension to account for added embedding vector
            total_input_dim = input_dim + embed_dim
        else:
            # If embedding not used, assume procedure_code is already included in input features
            total_input_dim = input_dim

        # One hidden layer
        self.hidden = nn.Linear(total_input_dim, hidden_dim)
        # Dropout regularization
        self.dropout = nn.Dropout(p)
        # batch normalization
        # self.bn = nn.BatchNorm1d(hidden_dim)
        # Output layer for main classification (num_classes outputs)
        self.output = nn.Linear(hidden_dim, num_classes)
        # Auxiliary head for uncertainty (single output for confidence)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        # initialize layer weights
        init.kaiming_normal_(self.hidden.weight)
        init.kaiming_normal_(self.output.weight)
        init.kaiming_normal_(self.confidence_head.weight, nonlinearity="sigmoid")

    def forward(
            self, 
            x: torch.Tensor, 
            high_card_codes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor of input features (excluding procedure_code if use_embedding is True, otherwise including all features).
        proc_code: Tensor of procedure code indices (if use_embedding is True). If use_embedding is False, this is ignored.
        """
        if self.use_embedding:
            # Look up embedding for procedure_code and concatenate with other features
            assert high_card_codes is not None, "high_card_codes must be provided when use_embedding is True"
            hc_embed = self.hc1_embedding(high_card_codes)        # shape: (batch_size, embed_dim)
            #hc_embed = hc_embed.squeeze(1) 
            x = torch.cat([x, hc_embed], dim=1)              # concatenate along feature dimension
        # Pass through hidden layer with non-linear activation
        h = F.elu(self.hidden(x))
        # pass throught Dropout
        h = self.dropout(h)
        # Pass through batch normalisation
        # h = self.bn(h)
        # Main classification output (logits for each class)
        logits = self.output(h)
        # Auxiliary confidence output (apply sigmoid to get probability between 0 and 1)
        confidence_logit = self.confidence_head(h)
        # confidence = torch.sigmoid(confidence_logit) # note: BCEWithLogitsLoss takes logits
        return logits, confidence_logit
