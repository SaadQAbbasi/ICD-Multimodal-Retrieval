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
            transforms.Resize(256), # Scale the image so that the shortest side = 256, keeping aspect ratio.
            transforms.CenterCrop(224), # Take a 224×224 center crop.
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    @torch.no_grad()
    def encode(self, image_paths):

        if isinstance(image_paths, str): # Accept either a single path or a list of paths
            image_paths = [image_paths]

        imgs = [self.transform(Image.open(image).convert("RGB")) for image in image_paths]
        x = torch.stack(imgs, dim=0)    # [batch_size, 3, 224, 224]

            
        device = next(self.model.parameters()).device      # Move input to model's device
        x = x.to(device)

        cls_embedding = self.model.get_intermediate_layers(x, n=1)[0][:, 0]  # CLS token
        cls_embedding = F.normalize(cls_embedding, dim=-1)  # Normalize

        if len(image_paths) == 1:
            return cls_embedding[0]  # return 1D tensor for single image
        
        return cls_embedding  # [batch_size, embedding_dim] for multiple images
