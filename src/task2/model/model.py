import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

import sys, os
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import torch.nn as nn
# from helpers.image_encoder import ImageEncoder      # or: from image_encoder import ImageEncoder
# from helpers.text_encoder import TextEncoder
# from helpers.fusion import FusionModule
# from helpers.projection import ProjectionModule


class ICDRetrievalModel(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        fusion_module,
        projection_module,
        icd_projection_module,
        embedding_dim,
        fusion_method
    ):
        """
        Complete model for ICD code retrieval from image and/or text queries.
        
        Args:
            image_encoder: ImageEncoder instance (frozen DINOv2)
            text_encoder: TextEncoder instance (medical BERT, shared for queries & ICDs)
            fusion_module: FusionModule for combining image and text embeddings into query embedding
            projection_module: ProjectionModule for query embeddings
            icd_projection_module: ProjectionModule for ICD text embeddings
            embedding_dim: Final common embedding space dimension same as projection module output dimension
            fusion_method: Method for fusion ("concat", "add", "multiply")
        """
        super().__init__()
        
        # Store encoders (not as nn.Module parameters since they're frozen/external)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        
        # Projection modules (trainable)
        self.query_projection = projection_module
        self.icd_projection = icd_projection_module
        
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
        
    def encode_image(self, image_path: str) -> torch.Tensor:
        """Extract image embedding using frozen DINO."""
        return self.image_encoder.encode(image_path) # returns normalized embedding of shape [batch_size, img_dim]
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Extract text embedding using BERT."""
        return self.text_encoder.encode(text) # returns normalized embedding of shape [batch_size, text_dim]
    
    def encode_icd_text(self, icd_description: str) -> torch.Tensor:
        """Extract ICD description embedding (uses same text encoder)."""
        return self.text_encoder.encode(icd_description) # returns normalized embedding of shape [batch_size, text_dim]

    def forward(
        self,
        image_paths: Optional[list[str]] = None,
        query_text: Optional[str] = None,
        icd_description: Optional[str] = None,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for one study with multiple images and one text report.

        Args:
            image_paths: List of image paths for this study
            query_text: Text report for the study
            icd_description: Optional ICD description for training
            return_all: Return intermediate embeddings

        Returns:
            Dict with:
                - query_embedding: [num_images, embedding_dim]
                - icd_embedding: [B, embedding_dim] (if provided)
                - optional raw/fused embeddings
        """
        outputs = {}

        # 1. Encode all images individually
        img_emb_list = []
        if image_paths is not None:
            for image in image_paths:
                img_emb = self.encode_image(image).squeeze(0) 
                img_emb_list.append(img_emb)

            img_emb = torch.stack(img_emb_list, dim=0)  # [num_images, img_dim]

            if return_all:
                outputs['raw_image_emb'] = img_emb
        else:
            img_emb = None

        # 2. Encode text once
        if query_text is not None:
            text_emb = self.encode_text(query_text).squeeze(0)  # [text_dim]

            if return_all:
                outputs['raw_text_emb'] = text_emb
        else:
            text_emb = None

        # 3. Fuse each image with text
        if img_emb is not None and text_emb is not None:
            # Expand text to match number of images
            text_exp = text_emb.unsqueeze(0).expand(img_emb.shape[0], -1)
            fused_emb = self.fusion_module.fuse(img_emb, text_exp)
        elif img_emb is not None:
            fused_emb = img_emb
        elif text_emb is not None:
            fused_emb = text_emb.unsqueeze(0)
        else:
            raise ValueError("At least one modality must be provided")

        if return_all:
            outputs['fused_emb'] = fused_emb

        # 4. Project fused embeddings to shared space
        query_embedding = self.query_projection(fused_emb)
        query_embedding = F.normalize(query_embedding, dim=-1)
        outputs['query_embedding'] = query_embedding

        # 5. ICD embeddings
        if icd_description is not None:
            icd_text_emb = self.encode_icd_text(icd_description)
            icd_embedding = self.icd_projection(icd_text_emb)
            icd_embedding = F.normalize(icd_embedding, dim=-1)
            outputs['icd_embedding'] = icd_embedding

        return outputs

