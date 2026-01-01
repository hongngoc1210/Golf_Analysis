"""
Neural Additive Model (NAM) Implementation
Score = β₀ + Σ fᵢ(xᵢ)

Mỗi feature có 1 NN riêng → Explainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNN(nn.Module):
    """
    Neural Network cho 1 feature duy nhất
    Input: xᵢ (scalar)
    Output: fᵢ(xᵢ) (scalar contribution)
    """
    def __init__(self, hidden_units=[64, 32], dropout=0.1):
        super().__init__()
        
        layers = []
        in_dim = 1  # Input là 1 feature scalar
        
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer: contribution của feature này
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x: (batch_size, 1) - giá trị của 1 feature
        output: (batch_size, 1) - contribution
        """
        return self.network(x)

class NAM(nn.Module):
    def __init__(self, num_features=17, hidden_units=[64, 32], dropout=0.1):
        super().__init__()
        self.num_features = num_features

        self.feature_nets = nn.ModuleList([
            FeatureNN(hidden_units, dropout)
            for _ in range(num_features)
        ])

        # Bias scalar (an toàn số học)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, return_contributions=False, clamp_output=False):
        contributions = []

        for i in range(self.num_features):
            x_i = x[:, i:i+1]
            contrib = self.feature_nets[i](x_i)
            contributions.append(contrib)

        contributions = torch.cat(contributions, dim=1)
        score = self.bias + contributions.sum(dim=1)

        # ❗ Clamp CHỈ khi inference
        if clamp_output:
            score = torch.clamp(score, 0.0, 10.0)

        if return_contributions:
            return score, contributions
        return score
    
    def get_feature_contributions(self, x):
        """
        Lấy contributions của từng feature (dùng cho XAI)
        
        Returns:
            dict: {feature_name: contribution_value}
        """
        with torch.no_grad():
            _, contributions = self.forward(x, return_contributions=True)
            return contributions.cpu().numpy()

class NAMLoss(nn.Module):
    def __init__(self, l2_lambda=1e-4, output_lambda=1e-3):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.output_lambda = output_lambda
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, model, contributions=None):
        """
        predictions: (batch,)
        targets: (batch,)
        contributions: (batch, num_features)
        """

        # 1. MSE
        mse_loss = self.mse(predictions, targets)

        # 2. L2 regularization (KHÔNG áp lên bias)
        l2_loss = 0.0
        for name, param in model.named_parameters():
            if "bias" not in name:
                l2_loss += torch.sum(param ** 2)
        l2_loss = self.l2_lambda * l2_loss

        # 3. Output regularization (trên batch thật)
        if contributions is not None:
            output_reg = self.output_lambda * torch.mean(contributions ** 2)
        else:
            output_reg = torch.tensor(0.0, device=predictions.device)

        total_loss = mse_loss + l2_loss + output_reg

        return {
            "total": total_loss,
            "mse": mse_loss,
            "l2": l2_loss,
            "output_reg": output_reg
        }
