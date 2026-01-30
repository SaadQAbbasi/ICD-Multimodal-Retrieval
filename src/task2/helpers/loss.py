import torch
import torch.nn.functional as F

def contrastive_loss(x_img, x_icd, temperature=0.07):
    """
    x_img: [B, D] fused image embedding (projected)
    x_icd: [B, D] ICD text embedding (projected)
    """
    # similarity matrix
    sim = x_img @ x_icd.T  # [B, B]
    sim = sim / temperature

    labels = torch.arange(x_img.size(0)).to(sim.device)
    loss_i2t = F.cross_entropy(sim, labels)      # image -> ICD
    loss_t2i = F.cross_entropy(sim.T, labels)    # ICD -> image
    return (loss_i2t + loss_t2i) / 2

