import torch
import torch.nn as nn

class FeatureNN(nn.Module):
    """
    Neural network for a single feature.
    Input : (batch, 1)
    Output: (batch, 1) → logit contribution
    """
    def __init__(self, hidden_units=(64, 32), dropout=0.1):
        super().__init__()

        layers = []
        in_dim = 1

        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NAMClassifier(nn.Module):
    """
    Neural Additive Model for Binary Classification
    logit = bias + sum_i f_i(x_i)
    """
    def __init__(self, num_features, hidden_units=(64, 32), dropout=0.1):
        super().__init__()

        self.num_features = num_features

        self.feature_nets = nn.ModuleList([
            FeatureNN(hidden_units, dropout)
            for _ in range(num_features)
        ])

        # Global bias (logit space)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_contributions=False):
        """
        Args:
            x: (batch, num_features)
        Returns:
            logits: (batch,)
            probs : (batch,)
            contributions (optional): (batch, num_features)
        """
        contribs = []

        for i in range(self.num_features):
            xi = x[:, i:i+1]
            ci = self.feature_nets[i](xi)
            contribs.append(ci)

        contributions = torch.cat(contribs, dim=1)
        logits = self.bias + contributions.sum(dim=1)
        probs = torch.sigmoid(logits)

        if return_contributions:
            return logits, probs, contributions

        return logits, probs

class NAMBinaryLoss(nn.Module):
    """
    Binary loss for NAM:
    L = BCEWithLogits + λ1 * L2(weights) + λ2 * mean(contrib^2)
    """
    def __init__(self, l2_lambda=1e-4, contrib_lambda=1e-3):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.l2_lambda = l2_lambda
        self.contrib_lambda = contrib_lambda

    def forward(self, logits, targets, model, contributions=None):
        """
        Args:
            logits: (batch,)
            targets: (batch,) ∈ {0,1}
            model: NAMClassifier
            contributions: (batch, num_features)
        """
        targets = targets.float()

        bce_loss = self.bce(logits, targets)

        l2_loss = torch.tensor(0.0, device=logits.device)

        for name, param in model.named_parameters():
            if param.requires_grad and "bias" not in name:
                l2_loss += torch.sum(param ** 2)

        l2_loss = self.l2_lambda * l2_loss
        if contributions is not None:
            contrib_reg = self.contrib_lambda * torch.mean(contributions ** 2)
        else:
            contrib_reg = torch.tensor(0.0, device=logits.device)

        total_loss = bce_loss + l2_loss + contrib_reg

        return {
            "total": total_loss,
            "bce": bce_loss,
            "l2": l2_loss,
            "contrib_reg": contrib_reg
        }
