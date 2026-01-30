# This class takes two an embedding and projects them to a desired dimension via various methods.

import torch
import torch.nn as nn

class ProjectionModule:
    def __init__(self, input_dim, output_dim, method="linear"):
        """
        input_dim: int, dimension of input embeddings
        output_dim: int, desired dimension of output embeddings
        method: str, one of ["linear", "mlp"]
        """
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

    def project(self, embedding):
        """
        embedding: tensor of shape [B, D_in]
        returns: projected embedding of shape [B, D_out]
        """
        return self.proj(embedding)