# This class takes two embeddings and fuses them in various ways
import torch
import torch.nn.functional as F

class FusionModule:
    def __init__(self, method="concat"):
        """
        method: str, one of ["concat", "add", "multiply"]
        """
        self.method = method

    def fuse(self, img_emb, text_emb):
        """
        img_emb: tensor of shape [B, D]
        text_emb: tensor of shape [B, D]
        returns: fused embedding of shape [B, D_fused]
        """
        if self.method == "concat":
            fused = torch.cat((img_emb, text_emb), dim=-1)
        elif self.method == "add":
            fused = img_emb + text_emb
        elif self.method == "multiply":
            fused = img_emb * text_emb
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        return fused