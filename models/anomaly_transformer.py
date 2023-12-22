"""
Module containing classes for Anomaly Attention and Anomaly Transformer models.

This module defines the AnomalyAttention and AnomalyTransformer classes,
which are used for attention-based anomaly detection in time series data.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from traceback import print_list


class AnomalyAttention(nn.Module):
    """
    Anomaly Attention module for attention-based anomaly detection.

    Attributes:
    - N (int): Number of elements in the sequence.
    - d_model (int): Dimensionality of the model.
    - device (str): Device on which the model is run.
    - Wq (nn.Linear): Linear transformation for query.
    - Wk (nn.Linear): Linear transformation for key.
    - Wv (nn.Linear): Linear transformation for value.
    - Ws (nn.Linear): Linear transformation for sigma.
    - Q (torch.Tensor): Query tensor.
    - K (torch.Tensor): Key tensor.
    - V (torch.Tensor): Value tensor.
    - sigma (torch.Tensor): Sigma tensor.
    - P (torch.Tensor): Prior association matrix.
    - S (torch.Tensor): Series association matrix.
    """

    def __init__(self, N, d_model, device):
        """
        Initialize the AnomalyAttention module.

        Parameters:
        - N (int): Number of elements in the sequence.
        - d_model (int): Dimensionality of the model.
        - device (str): Device on which the model is run.
        """

        super(AnomalyAttention, self).__init__()

        self.d_model = d_model
        self.N = N
        self.device = device

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)

        self.Q = self.K = self.V = self.sigma = torch.zeros((N, d_model))

        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

    def forward(self, x):
        """
        Forward pass of the AnomalyAttention module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """

        self.initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        Z = self.reconstruction()

        return Z.squeeze(2)

    def initialize(self, x):
        """
        Initialize the tensors for attention computation.

        Parameters:
        - x (torch.Tensor): Input tensor.
        """

        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)

        self.sigma = self.Ws(x)
        self.sigma = torch.sigmoid(self.sigma * 5) + 1e-5
        self.sigma = torch.pow(3, self.sigma) - 1

    @staticmethod
    def gaussian_kernel(mean, sigma, device):
        """
        Compute a Gaussian kernel.

        Parameters:
        - mean (torch.Tensor): Mean tensor.
        - sigma (torch.Tensor): Sigma tensor.
        - device (str): Device on which the model is run.

        Returns:
        - torch.Tensor: Gaussian kernel.
        """

        normalize = (1 / (math.sqrt(2 * torch.pi) * sigma)).to(device)
        
        return normalize * torch.exp(-0.5 * (mean.to(device) / sigma.to(device)).pow(2))

    def prior_association(self):
        """
        Compute the prior association matrix.

        Returns:
        - torch.Tensor: Prior association matrix.
        """

        p = torch.from_numpy(
            np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1])
        )

        gaussian = self.gaussian_kernel(p.float(), self.sigma, self.device)
        gaussian /= gaussian.sum(dim=(2,1)).unsqueeze(1).unsqueeze(2)

        return gaussian

    def series_association(self):
        """
        Compute the series association matrix.

        Returns:
        - torch.Tensor: Series association matrix.
        """
        return F.softmax(torch.bmm(self.Q, self.K.transpose(1,2)) / math.sqrt(self.d_model), dim=0)

    def reconstruction(self):
        """
        Perform the reconstruction step.

        Returns:
        - torch.Tensor: Reconstructed tensor.
        """
        return self.S @ self.V


class AnomalyTransformerBlock(nn.Module):
    """
    Anomaly Transformer Block for attention-based anomaly detection.

    Attributes:
    - N (int): Number of elements in the sequence.
    - d_model (int): Dimensionality of the model.
    - attention (AnomalyAttention): AnomalyAttention module.
    - ln1 (nn.LayerNorm): Layer normalization for the first step.
    - ff (nn.Sequential): Feedforward module.
    - ln2 (nn.LayerNorm): Layer normalization for the second step.
    """

    def __init__(self, N, d_model, device):
        """
        Initialize the AnomalyTransformerBlock.

        Parameters:
        - N (int): Number of elements in the sequence.
        - d_model (int): Dimensionality of the model.
        - device (str): Device on which the model is run.
        """

        super().__init__()

        self.N, self.d_model = N, d_model
        self.attention = AnomalyAttention(self.N, self.d_model, device=device)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        """
        Forward pass of the AnomalyTransformerBlock.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """

        x_identity = x
        x = self.attention(x)
        z = self.ln1(x + x_identity)

        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        return z


class AnomalyTransformer(nn.Module):
    """
    Anomaly Transformer model for attention-based anomaly detection.

    Attributes:
    - N (int): Number of elements in the sequence.
    - d_model (int): Dimensionality of the model.
    - layers (int): Number of transformer blocks.
    - lambda_ (float): Hyperparameter for association discrepancy.
    - device (str): Device on which the model is run.
    - blocks (nn.ModuleList): List of AnomalyTransformerBlock modules.
    - output (None): Output placeholder.
    - classifier (nn.Linear): Linear layer for classification.
    - P_layers (list): List of prior association matrices for each block.
    - S_layers (list): List of series association matrices for each block.
    """

    def __init__(self, N, d_model, layers, lambda_, device):
        """
        Initialize the AnomalyTransformer.

        Parameters:
        - N (int): Number of elements in the sequence.
        - d_model (int): Dimensionality of the model.
        - layers (int): Number of transformer blocks.
        - lambda_ (float): Hyperparameter for association discrepancy.
        - device (str): Device on which the model is run.
        """

        super().__init__()

        self.N = N
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model, device) for _ in range(layers)]
        )
        self.output = None
        self.lambda_ = lambda_
        self.classifier = nn.Linear(d_model, 1)

        self.P_layers = []
        self.S_layers = []

    def forward(self, x):
        """
        Forward pass of the AnomalyTransformer.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """

        self.P_layers = []
        self.S_layers = []

        for block in self.blocks:

            x = block(x)
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)

        x = self.classifier(x)

        return x.squeeze(2)

    def layer_association_discrepancy(self, Pl, Sl):
        """
        Compute the association discrepancy for a single layer.

        Parameters:
        - Pl (torch.Tensor): Prior association matrix.
        - Sl (torch.Tensor): Series association matrix.

        Returns:
        - torch.Tensor: Association discrepancy vector.
        """

        rowwise_kl = lambda row: (
            F.kl_div(Pl[row, :], Sl[row, :]) + F.kl_div(Sl[row, :], Pl[row, :])
        )

        ad_vector = torch.Tensor([rowwise_kl(row).item() for row in range(len(Pl))])
        
        return ad_vector

    def association_discrepancy(self, P_list, S_list):
        """
        Compute the overall association discrepancy.

        Parameters:
        - P_list (list): List of prior association matrices.
        - S_list (list): List of series association matrices.

        Returns:
        - torch.Tensor: Overall association discrepancy vector.
        """

        return (1 / len(P_list)) * sum(
            [
                self.layer_association_discrepancy(P, S)
                for P, S in zip(P_list, S_list)
            ]
        )

    def loss_function(self, x_hat, P_list, S_list, lambda_, x):
        """
        Compute the loss function.

        Parameters:
        - x_hat (torch.Tensor): Predicted output tensor.
        - P_list (list): List of prior association matrices.
        - S_list (list): List of series association matrices.
        - lambda_ (float): Hyperparameter for association discrepancy.
        - x (torch.Tensor): Ground truth tensor.

        Returns:
        - torch.Tensor: Loss value.
        """

        mse_loss = F.mse_loss(x_hat, x)
        assoc_discrepancy = self.association_discrepancy(P_list, S_list)
        assoc_discrepancy_absmean = torch.mean(torch.abs(assoc_discrepancy))
        
        return mse_loss - (lambda_ * assoc_discrepancy_absmean)

    def min_loss(self, preds, y):
        """
        Compute the minimum loss.

        Parameters:
        - preds (torch.Tensor): Model predictions.
        - y (torch.Tensor): Ground truth.

        Returns:
        - torch.Tensor: Minimum loss value.
        """

        P_list = self.P_layers
        S_list = [S.detach() for S in self.S_layers]
        lambda_ = -self.lambda_
        
        return self.loss_function(preds, P_list, S_list, lambda_, y)

    def max_loss(self, preds, y):
        """
        Compute the maximum loss.

        Parameters:
        - preds (torch.Tensor): Model predictions.
        - y (torch.Tensor): Ground truth.

        Returns:
        - torch.Tensor: Maximum loss value.
        """

        P_list = [P.detach() for P in self.P_layers]
        S_list = self.S_layers
        lambda_ = self.lambda_
        
        return self.loss_function(preds, P_list, S_list, lambda_, y)

    def loss_fn(self, preds, y):
        """
        Compute the loss function for training.

        Parameters:
        - preds (torch.Tensor): Model predictions.
        - y (torch.Tensor): Ground truth.

        Returns:
        - torch.Tensor: Loss value.
        """

        loss = self.min_loss(preds, y)
        loss.backward(retain_graph=True)
        loss = self.max_loss(preds, y)

        return loss
