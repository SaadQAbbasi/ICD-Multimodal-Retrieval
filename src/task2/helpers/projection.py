# This class takes two an embedding and projects them to a desired dimension via various methods.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionModule(nn.Module):
    def __init__(self, input_dim, output_dim, method="linear"):
        """
        input_dim: int, dimension of input embeddings
        output_dim: int, desired dimension of output embeddings
        method: str, one of ["linear", "mlp"]
        """
        super().__init__()
        self.method = method
        if method == "linear":
            self.proj = nn.Linear(input_dim, output_dim)
        elif method == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(input_dim, (input_dim + output_dim) // 2),
                nn.ReLU(),
                nn.Linear((input_dim + output_dim) // 2, output_dim)
            )
        else:
            raise ValueError(f"Unknown projection method: {method}")

    def forward(self, embedding):
        """
        embedding: tensor of shape [B, D_in]
        returns: projected embedding of shape [B, D_out]
        """

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)     # ensure batch dimension

        embedding = embedding.to(next(self.proj.parameters()).device)       # move to module's device

        projected = self.proj(embedding)
        projected = F.normalize(projected, dim=-1)  # Normalize to unit length

        return projected
    
