"""
Training script for multimodal radiograph retrieval with InfoNCE contrastive loss.

This implements the full pipeline:
1. Encode images with DINO
2. Encode text with MedCLIP
3. Fuse image + text embeddings
4. Compute InfoNCE loss (pull same-study embeddings together, push different-study apart)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import wandb  # Optional: for logging
import sys
sys.path.append("..")   # path to your project root


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for contrastive learning.
    
    Pulls embeddings from the same study together, pushes embeddings from 
    different studies apart.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        study_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            embeddings: [N, D] tensor of normalized embeddings
            study_indices: [N] tensor indicating which study each embedding belongs to
            
        Returns:
            Scalar loss
        """
        # Compute similarity matrix
        # [N, D] @ [D, N] = [N, N]
        similarity = embeddings @ embeddings.T / self.temperature
        
        # Create positive mask: embeddings from same study
        # [N, 1] == [1, N] -> [N, N]
        positive_mask = (study_indices.unsqueeze(1) == study_indices.unsqueeze(0)).float()
        
        # Remove self-similarities (diagonal)
        positive_mask.fill_diagonal_(0)
        
        # Create negative mask: embeddings from different studies
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # For each anchor, we want to:
        # - Maximize similarity to positives (same study)
        # - Minimize similarity to negatives (different studies)
        
        # Numerator: exp(sim(anchor, positive))
        # Denominator: sum of exp(sim(anchor, all others except self))
        
        # Calculate loss for each anchor
        exp_sim = torch.exp(similarity)
        
        # Sum over positives
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        
        # Sum over all (excluding self)
        all_sum = (exp_sim * (1 - torch.eye(embeddings.shape[0], device=embeddings.device))).sum(dim=1)
        
        # InfoNCE: -log(sum(positives) / sum(all))
        # Add small epsilon to avoid log(0)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Only compute loss for anchors that have positives
        has_positive = (positive_mask.sum(dim=1) > 0)
        
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return loss[has_positive].mean()


class MultimodalContrastiveTrainer:
    """
    Trainer for multimodal radiograph retrieval with contrastive learning.
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        temperature: float = 0.07,
        log_wandb: bool = False,
    ):
        """
        Args:
            model: ICDRetrievalModel instance
            train_loader: Training dataloader (use flat collate)
            val_loader: Validation dataloader (use flat collate)
            optimizer: Optimizer
            device: Device to train on
            temperature: Temperature for InfoNCE loss
            log_wandb: Whether to log to wandb
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.log_wandb = log_wandb
        
        # Loss function
        self.criterion = InfoNCELoss(temperature=temperature)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # batch contains (with flat collate):
            # - study_ids: List[str] - one per image
            # - images: List of image paths - flattened
            # - report_texts: List[str] - repeated per image
            # - study_indices: Tensor[N] - which study each image belongs to
            # - images_per_study: List[int] - number of images per study
            
            self.optimizer.zero_grad()
            
            # Process batch
            loss = self._forward_batch(batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss
        }
    
    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass for one batch with contrastive loss.
        
        Strategy:
        1. Encode each image with DINO
        2. Encode each text with MedCLIP (one per study, but repeated for each image belonging to the same study)
        3. Create fused embeddings for each image-text pair
        4. Gather all embeddings (image, text, fused) into one pool
        5. Compute InfoNCE loss
        
        Args:
            batch: Batch from dataloader with flat collate
            
        Returns:
            Contrastive loss
        """
        image_paths = batch['images']  # List of paths
        report_texts = batch['report_texts']  # List of texts (repeated)
        study_indices = batch['study_indices'].to(self.device)  # Tensor[N]
        
        # Storage for all embeddings
        all_embeddings = []
        all_study_labels = []
        
        # Process each image-text pair
        for idx, (img_path, text) in enumerate(zip(image_paths, report_texts)):
            study_idx = study_indices[idx].item()
            
            # Forward through model
            # This returns image, text, and fused embeddings for ONE study
            outputs = self.model(
                image_paths=[img_path],  # Single image as list
                query_text=text,
                return_all=False
            )
            
            # Get query embedding (this is the fused embedding)
            query_emb = outputs['query_embedding']  # [1, D]
            
            all_embeddings.append(query_emb.squeeze(0))
            all_study_labels.append(study_idx)
        
        # Stack all embeddings
        embeddings = torch.stack(all_embeddings)  # [N, D]
        study_labels = torch.tensor(all_study_labels, device=self.device)  # [N]
        
        # Ensure embeddings are normalized
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute contrastive loss
        loss = self.criterion(embeddings, study_labels)
        
        return loss
    
    def _forward_batch_optimized(self, batch: Dict) -> torch.Tensor:
        """
        OPTIMIZED version: Process images in parallel.
        
        This version is faster but requires more memory.
        Use this if you have enough GPU memory.
        """
        image_paths = batch['images']
        report_texts = batch['report_texts']
        study_indices = batch['study_indices'].to(self.device)
        
        # Get unique studies and their texts
        unique_studies = study_indices.unique()
        study_to_text = {}
        
        for study_idx in unique_studies:
            mask = study_indices == study_idx
            # Get first text for this study (they're all the same)
            study_to_text[study_idx.item()] = report_texts[mask.nonzero()[0].item()]
        
        # Batch encode images
        image_embeddings = []
        for img_path in image_paths:
            img_emb = self.model.encode_image(img_path)
            image_embeddings.append(img_emb.squeeze(0))
        
        image_embeddings = torch.stack(image_embeddings)  # [N, img_dim]
        
        # Encode texts (one per unique study)
        text_embeddings_dict = {}
        for study_idx, text in study_to_text.items():
            text_emb = self.model.encode_text(text)
            text_embeddings_dict[study_idx] = text_emb.squeeze(0)
        
        # Match text embeddings to images
        text_embeddings = []
        for study_idx in study_indices:
            text_embeddings.append(text_embeddings_dict[study_idx.item()])
        
        text_embeddings = torch.stack(text_embeddings)  # [N, text_dim]
        
        # Fuse
        fused_embeddings = self.model.fusion_module.fuse(image_embeddings, text_embeddings)
        
        # Project
        query_embeddings = self.model.query_projection(fused_embeddings)
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        
        # Compute loss
        loss = self.criterion(query_embeddings, study_indices)
        
        return loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Validation {epoch}")
        
        for batch in progress_bar:
            # Forward pass
            loss = self._forward_batch(batch)
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'avg_val_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        
        return {
            'val_loss': avg_loss
        }
    
    def train(
        self, 
        num_epochs: int,
        save_dir: str = './checkpoints',
        save_every: int = 5
    ):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")

            # # Scheduler step (if using scheduler)
            # if hasattr(self.optimizer, 'scheduler'):
            #     self.optimizer.scheduler.step()
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Log to wandb
            if self.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['train_loss'],
                    'val/loss': val_metrics['val_loss']
                })
            
            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, best_path)
                print(f"New best model! Val loss: {best_val_loss:.4f}")


def main():
    """
    Example training script.
    """
    import sys
    sys.path.append('..')
    
    from data.dataset import create_dataloader
    from torchvision import transforms
    
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    TEMPERATURE = 0.07
    MIN_IMAGES_PER_STUDY = 2
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataloaders with FLAT collate
    print("Loading datasets...")
    train_loader = create_dataloader(
        mimic_root="/path/to/mimic-cxr-jpg",  # UPDATE THIS
        split='train',
        batch_size=BATCH_SIZE,
        min_images_per_study=MIN_IMAGES_PER_STUDY,
        num_workers=4,
        transform=transform,
        return_all_images=True,
        use_flat_collate=True,  # IMPORTANT: Use flat collate for contrastive learning
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        mimic_root="/path/to/mimic-cxr-jpg",  # UPDATE THIS
        split='val',
        batch_size=BATCH_SIZE,
        min_images_per_study=MIN_IMAGES_PER_STUDY,
        num_workers=4,
        transform=transform,
        return_all_images=True,
        use_flat_collate=True,  # IMPORTANT: Use flat collate
        shuffle=False,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model components
    from helpers.image_encoder import ImageEncoder
    from helpers.text_encoder import TextEncoder
    from helpers.fusion import FusionModule
    from helpers.projection import ProjectionModule
    
    # Example dimensions (adjust to your actual models)
    IMG_DIM = 768  # DINO output dimension
    TEXT_DIM = 768  # MedCLIP output dimension
    EMBEDDING_DIM = 512  # Final embedding dimension
    
    image_encoder = ImageEncoder()  # Your DINO encoder
    text_encoder = TextEncoder()    # Your MedCLIP encoder
    fusion_module = FusionModule(
        image_dim=IMG_DIM,
        text_dim=TEXT_DIM,
        fusion_method='concat'  # or 'add', 'multiply'
    )
    projection_module = ProjectionModule(
        input_dim=fusion_module.output_dim,
        output_dim=EMBEDDING_DIM
    )
    
    # We don't use ICD projection for retrieval, but keep it for compatibility
    icd_projection_module = ProjectionModule(
        input_dim=TEXT_DIM,
        output_dim=EMBEDDING_DIM
    )
    
    # Create model
    from model import ICDRetrievalModel  # Your model file
    
    model = ICDRetrievalModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion_module=fusion_module,
        projection_module=projection_module,
        icd_projection_module=icd_projection_module,
        embedding_dim=EMBEDDING_DIM,
        fusion_method='concat'
    )
    
    # Optimizer (only train projection and fusion modules, encoders are frozen)
    trainable_params = list(fusion_module.parameters()) + \
                      list(projection_module.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # # Scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=NUM_EPOCHS,
    #     eta_min=1e-6
    # )
    
    # Initialize wandb (optional)
    wandb.init(
        project="radiograph-retrieval",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "num_epochs": NUM_EPOCHS,
        }
    )
    
    # Create trainer
    trainer = MultimodalContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        temperature=TEMPERATURE,
        log_wandb=True,  # Set to True if using wandb
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir='./checkpoints',
        save_every=5
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()