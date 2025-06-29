import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
import pandas as pd
from opacus import PrivacyEngine  # Import Opacus


class PytorchModel(nn.Module, BaseEstimator):
    def __init__(self, random_state=42, hidden_sizes=None, batch_size=None, epochs=None,
                 learning_rate=None, lr_factor=0.1, use_dp=False, 
                 max_grad_norm=1.0, use_dropout=False, dropout_p=0.5, weight_decay=0.0,
                 target_epsilon=1.0, target_delta=1e-5):
        """
        Args:
            use_dp: If True, train with differential privacy using Opacus.
            max_grad_norm: Maximum gradient norm for DP clipping.
            use_dropout: If True, insert dropout layers.
            dropout_p: Dropout probability.
            weight_decay: L2 regularization coefficient.
            target_epsilon: Desired privacy budget (ε). Training stops early if reached.
            target_delta: δ for privacy accounting.
        """
        super().__init__()
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCELoss(reduction="none")
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_dp = use_dp
        self.max_grad_norm = max_grad_norm
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.output_shape = 1

        self.lr_factor = 0.1 if lr_factor is None else lr_factor

    def build_network(self, input_size, hidden_sizes=None):
        if hidden_sizes is None:
            hidden_sizes = [200, 100]

        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            if self.use_dropout:
                layers.append(nn.Dropout(p=self.dropout_p))
        layers.append(nn.Linear(layer_sizes[-1], self.output_shape))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        return self.model(x)
        
    def fit(self, X_train, y_train, epochs=None, batch_size=None, learning_rate=None,
            sample_weight=None, hidden_sizes=None, desc="Training Progress"):
        # Build the network as usual.
        self.build_network(input_size=X_train.shape[1], hidden_sizes=hidden_sizes)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes

        # Convert input data.
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        self.classes_ = np.unique(y_train.cpu().numpy().ravel())

        if sample_weight is not None:
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32).to(self.device)
            indices = torch.randperm(X_train.size(0))
            X_train = X_train[indices]
            y_train = y_train[indices]
            sample_weight = sample_weight[indices]

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create the optimizer.
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.weight_decay)

        # If using DP, wrap the training objects using make_private_with_epsilon
        if self.use_dp:
            # Use just the default RDP accountant
            privacy_engine = PrivacyEngine()
            
            # Call make_private_with_epsilon
            dp_module, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                module=self,
                optimizer=optimizer,
                data_loader=dataloader,
                max_grad_norm=self.max_grad_norm,
                target_delta=self.target_delta,
                target_epsilon=self.target_epsilon,
                epochs=epochs
            )
            
            # Store the privacy engine
            self.privacy_engine = privacy_engine

        # Save lr_factor locally in case it's needed later.
        local_lr_factor = self.lr_factor
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            factor=local_lr_factor, patience=10)
        progress_bar = tqdm(range(epochs), desc=desc, unit="epoch")
        for epoch in progress_bar:
            self.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                if sample_weight is not None:
                    batch_start = batch_idx * batch_size
                    batch_end = batch_start + inputs.shape[0]
                    batch_instance_weight = sample_weight[batch_start:batch_end]
                    loss = (loss * batch_instance_weight.view(-1, 1)).mean()
                else:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            scheduler.step(running_loss / len(dataloader))
            postfix = {"Loss": f"{running_loss/len(dataloader):.4f}"}
            
            # Simplify the epsilon reporting to just use the default accountant
            if self.use_dp:
                try:
                    epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
                    postfix["ε"] = f"{epsilon:.2f}"  
                except Exception:
                    # In case calculation fails, don't crash the training
                    pass
                    
            progress_bar.set_postfix(postfix)
        progress_bar.close()

    def predict(self, X):
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self(X)
            predicted = (outputs >= 0.5).int().cpu().numpy().ravel()
        return predicted

    def predict_proba(self, X):
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self(X).cpu().numpy().ravel()
        probs = np.zeros((outputs.shape[0], 2))
        probs[:, 1] = outputs
        probs[:, 0] = 1 - outputs
        return probs

    def compute_loss(self, X, y):
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        with torch.no_grad():
            outputs = self(X)
            loss = self.criterion(outputs, y)
        return loss

    def get_params(self, deep=True):
        return {
            "hidden_sizes": self.hidden_sizes,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr_factor": self.lr_factor,
            "use_dp": self.use_dp,
            "max_grad_norm": self.max_grad_norm,
            "use_dropout": self.use_dropout,
            "dropout_p": self.dropout_p,
            "weight_decay": self.weight_decay,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_penultimate_representation_(self, x):
        with torch.no_grad():
            return self.model[:-1](x)

    def get_intermediate_activations_(self, x, layer_indices):
        activations = []
        current = x
        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                current = layer(current)
                if idx in layer_indices:
                    activations.append(current)
        return activations

    def get_intermediate_activations(self, x, layer_indices):
        activations = []
        current = x
        for idx, (name, layer) in enumerate(self.model.named_children()):
            current = layer(current)
            if idx in layer_indices:
                activations.append(current)
        return activations

    def get_penultimate_representation(self, x):
        for name, module in list(self.model.named_children())[:-1]:
            x = module(x)
        return x


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def build_pytorch_model_for_seed(trainer):
    model = PytorchModel(
        random_state=trainer.seed,
        hidden_sizes=trainer.fit_params.get("hidden_sizes", [200, 100, 50]),
        batch_size=trainer.fit_params.get("batch_size", 256),
        epochs=trainer.fit_params.get("epochs", 100),
        learning_rate=trainer.fit_params.get("learning_rate", 0.002),
        lr_factor=trainer.fit_params.get("lr_factor", 0.1),
        use_dp=trainer.fit_params.get("use_dp", False),
        max_grad_norm=trainer.fit_params.get("max_grad_norm", 1.0),
        use_dropout=trainer.fit_params.get("use_dropout", False),
        dropout_p=trainer.fit_params.get("dropout_p", 0.5),
        weight_decay=trainer.fit_params.get("weight_decay", 0.0),
        target_epsilon=trainer.fit_params.get("target_epsilon", 1.0),
        target_delta=trainer.fit_params.get("target_delta", 1e-5),
    ).to(trainer.device)
    model.classes_ = np.unique(trainer.data.labels.ravel())
    input_size = trainer.data.features.shape[1]
    hidden_sizes = trainer.fit_params.get("hidden_sizes", [200, 100, 50])
    model.build_network(input_size=input_size, hidden_sizes=hidden_sizes)
    return model
