# image_encoder.py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class ImageEncoder:
    def __init__(self, model):
        """
        model: pretrained DINO backbone (ViT)
        """
        self.model = model
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    @torch.no_grad()
    def encode(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)  # [1,3,224,224]
        cls_embedding = self.model.get_intermediate_layers(x, n=1)[0][:, 0]
        cls_embedding = F.normalize(cls_embedding, dim=-1)
        return cls_embedding
