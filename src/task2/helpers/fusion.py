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
        img_emb: tensor of shape [B, D_img]
        text_emb: tensor of shape [B, D_text]
        returns: fused embedding of shape [B, D_fused]
        """


        # If input is a single embedding (1D), add batch dimension to make it [1, D]
        if img_emb.dim() == 1: 
            img_emb = img_emb.unsqueeze(0) 
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0) 


        # move text_emb to img_emb device
        text_emb = text_emb.to(img_emb.device)

        if self.method == "concat":
            fused = torch.cat((img_emb, text_emb), dim=-1) # [B, D_img + D_text]
        elif self.method == "add":
            if img_emb.shape[-1] != text_emb.shape[-1]:
                raise ValueError("For 'add' fusion, image and text embeddings must have the same dimension")
            fused = img_emb + text_emb # [B, D] where D = D_img = D_text
        elif self.method == "multiply":
            if img_emb.shape[-1] != text_emb.shape[-1]:
                raise ValueError("For 'multiply' fusion, image and text embeddings must have the same dimension")
            fused = img_emb * text_emb # [B, D] where D = D_img = D_text
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        return fused