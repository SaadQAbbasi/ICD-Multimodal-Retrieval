"""
Evaluation script for multimodal radiograph retrieval.

Implements Recall@K metric:
- For each query (image or text), retrieve top-K most similar images
- Success if any retrieved image is from the same study
- Recall@K = proportion of successful retrievals
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append("..")   # path to your project root


class RetrievalEvaluator:
    """
    Evaluator for multimodal radiograph retrieval.
    
    Computes Recall@K metric on validation/test set.
    """
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
        k_values: List[int] = [1, 5, 10, 20],
    ):
        """
        Args:
            model: Trained ICDRetrievalModel
            device: Device to run on
            k_values: List of K values for Recall@K
        """
        self.model = model.to(device)
        self.device = device
        self.k_values = k_values
        
    @torch.no_grad()
    def build_database(self, dataloader: DataLoader) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Build embedding database from all images in the dataset.
        
        Args:
            dataloader: DataLoader (flat collate)
            
        Returns:
            embeddings: [N, D] tensor of all image embeddings
            study_ids: List of study IDs for each embedding
            image_paths: List of image paths for each embedding
        """
        self.model.eval()
        
        all_embeddings = []
        all_study_ids = []
        all_image_paths = []
        
        print("Building embedding database...")
        for batch in tqdm(dataloader):
            image_paths = batch['images']
            report_texts = batch['report_texts']
            study_ids = batch['study_ids']
            
            # Encode each image
            for img_path, text, study_id in zip(image_paths, report_texts, study_ids):
                # Forward through model
                outputs = self.model(
                    image_paths=[img_path],
                    query_text=text,
                    return_all=False
                )
                
                embedding = outputs['query_embedding'].squeeze(0)  # [D]
                
                all_embeddings.append(embedding)
                all_study_ids.append(study_id)
                all_image_paths.append(img_path)
        
        embeddings = torch.stack(all_embeddings)  # [N, D]
        embeddings = F.normalize(embeddings, dim=-1)  # Normalize
        
        return embeddings, all_study_ids, all_image_paths
    
    @torch.no_grad()
    def evaluate_image_to_image(
        self,
        embeddings: torch.Tensor,
        study_ids: List[str],
        image_paths: List[str],
    ) -> Dict[int, float]:
        """
        Evaluate image-to-image retrieval.
        
        For each image, use it as a query to retrieve other images.
        Success if any retrieved image is from the same study.
        
        Args:
            embeddings: [N, D] database embeddings
            study_ids: Study ID for each embedding
            image_paths: Image path for each embedding
            
        Returns:
            Dict mapping K -> Recall@K
        """
        print("\nEvaluating Image-to-Image Retrieval...")
        
        N = embeddings.shape[0]
        recalls = {k: 0.0 for k in self.k_values}
        num_queries = 0
        
        # For each image as a query
        for query_idx in tqdm(range(N)):
            query_emb = embeddings[query_idx:query_idx+1]  # [1, D]
            query_study = study_ids[query_idx]
            query_path = image_paths[query_idx]
            
            # Compute similarity to all other images
            similarities = (query_emb @ embeddings.T).squeeze(0)  # [N]
            
            # Sort by similarity (descending)
            sorted_indices = similarities.argsort(descending=True)
            
            # Skip the query image itself
            sorted_indices = sorted_indices[sorted_indices != query_idx]
            
            # Check each K value
            for k in self.k_values:
                # Get top-K retrieved images
                top_k_indices = sorted_indices[:k]
                
                # Check if any retrieved image is from the same study
                retrieved_studies = [study_ids[idx.item()] for idx in top_k_indices]
                
                # Success if any retrieved image is from same study
                if query_study in retrieved_studies:
                    recalls[k] += 1.0
            
            num_queries += 1
        
        # Compute average recall
        for k in self.k_values:
            recalls[k] /= num_queries
        
        return recalls
    
    @torch.no_grad()
    def evaluate_text_to_image(
        self,
        dataloader: DataLoader,
        database_embeddings: torch.Tensor,
        database_study_ids: List[str],
    ) -> Dict[int, float]:
        """
        Evaluate text-to-image retrieval.
        
        For each study's text report, use it to retrieve images.
        Success if any retrieved image is from the same study.
        
        Args:
            dataloader: DataLoader for queries
            database_embeddings: [N, D] database embeddings
            database_study_ids: Study ID for each database embedding
            
        Returns:
            Dict mapping K -> Recall@K
        """
        print("\nEvaluating Text-to-Image Retrieval...")
        
        self.model.eval()
        
        recalls = {k: 0.0 for k in self.k_values}
        num_queries = 0
        
        # Group by study (since text is repeated for each image)
        processed_studies = set()
        
        for batch in tqdm(dataloader):
            study_ids = batch['study_ids']
            report_texts = batch['report_texts']
            
            for study_id, text in zip(study_ids, report_texts):
                # Skip if already processed this study
                if study_id in processed_studies:
                    continue
                
                processed_studies.add(study_id)
                
                # Encode text query
                text_emb = self.model.encode_text(text)  # [1, text_dim]
                query_emb = self.model.query_projection(text_emb)
                query_emb = F.normalize(query_emb, dim=-1)  # [1, D]
                
                # Compute similarity to all images in database
                similarities = (query_emb @ database_embeddings.T).squeeze(0)  # [N]
                
                # Sort by similarity
                sorted_indices = similarities.argsort(descending=True)
                
                # Check each K value
                for k in self.k_values:
                    top_k_indices = sorted_indices[:k]
                    
                    # Check if any retrieved image is from the same study
                    retrieved_studies = [database_study_ids[idx.item()] for idx in top_k_indices]
                    
                    if study_id in retrieved_studies:
                        recalls[k] += 1.0
                
                num_queries += 1
        
        # Compute average recall
        for k in self.k_values:
            recalls[k] /= num_queries
        
        return recalls
    
    @torch.no_grad()
    def evaluate_multimodal_to_image(
        self,
        dataloader: DataLoader,
        database_embeddings: torch.Tensor,
        database_study_ids: List[str],
        database_image_paths: List[str],
    ) -> Dict[int, float]:
        """
        Evaluate multimodal (image+text) to image retrieval.
        
        For each image+text pair, use the fused embedding to retrieve images.
        Success if any retrieved image (excluding query) is from the same study.
        
        Args:
            dataloader: DataLoader for queries
            database_embeddings: [N, D] database embeddings
            database_study_ids: Study ID for each database embedding
            database_image_paths: Image path for each database embedding
            
        Returns:
            Dict mapping K -> Recall@K
        """
        print("\nEvaluating Multimodal (Image+Text) to Image Retrieval...")
        
        self.model.eval()
        
        recalls = {k: 0.0 for k in self.k_values}
        num_queries = 0
        
        for batch in tqdm(dataloader):
            image_paths = batch['images']
            report_texts = batch['report_texts']
            study_ids = batch['study_ids']
            
            for img_path, text, study_id in zip(image_paths, report_texts, study_ids):
                # Encode multimodal query (image + text fused)
                outputs = self.model(
                    image_paths=[img_path],
                    query_text=text,
                    return_all=False
                )
                
                query_emb = outputs['query_embedding']  # [1, D]
                
                # Compute similarity to all images in database
                similarities = (query_emb @ database_embeddings.T).squeeze(0)  # [N]
                
                # Sort by similarity
                sorted_indices = similarities.argsort(descending=True)
                
                # Find and skip the query image itself in the database
                query_img_idx = None
                for idx, db_path in enumerate(database_image_paths):
                    if db_path == img_path:
                        query_img_idx = idx
                        break
                
                if query_img_idx is not None:
                    # Remove query image from results
                    mask = torch.ones(len(sorted_indices), dtype=torch.bool)
                    mask[sorted_indices == query_img_idx] = False
                    sorted_indices = sorted_indices[mask]
                
                # Check each K value
                for k in self.k_values:
                    top_k_indices = sorted_indices[:k]
                    
                    # Check if any retrieved image is from the same study
                    retrieved_studies = [database_study_ids[idx.item()] for idx in top_k_indices]
                    
                    if study_id in retrieved_studies:
                        recalls[k] += 1.0
                
                num_queries += 1
        
        # Compute average recall
        for k in self.k_values:
            recalls[k] /= num_queries
        
        return recalls
    
    def evaluate_all(self, dataloader: DataLoader) -> Dict[str, Dict[int, float]]:
        """
        Run all evaluation metrics.
        
        Args:
            dataloader: DataLoader with flat collate
            
        Returns:
            Dict with results for each retrieval mode
        """
        # Build database
        embeddings, study_ids, image_paths = self.build_database(dataloader)
        
        print(f"\nDatabase size: {len(embeddings)} images")
        print(f"Unique studies: {len(set(study_ids))}")
        
        results = {}
        
        # 1. Image-to-Image Retrieval
        img_to_img = self.evaluate_image_to_image(embeddings, study_ids, image_paths)
        results['image_to_image'] = img_to_img
        
        # 2. Text-to-Image Retrieval
        text_to_img = self.evaluate_text_to_image(dataloader, embeddings, study_ids)
        results['text_to_image'] = text_to_img
        
        # 3. Multimodal-to-Image Retrieval
        multimodal_to_img = self.evaluate_multimodal_to_image(
            dataloader, embeddings, study_ids, image_paths
        )
        results['multimodal_to_image'] = multimodal_to_img
        
        return results
    
    def print_results(self, results: Dict[str, Dict[int, float]]):
        """Pretty print evaluation results."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        for mode, recalls in results.items():
            print(f"\n{mode.upper().replace('_', ' ')}:")
            for k, recall in sorted(recalls.items()):
                print(f"  Recall@{k:2d}: {recall*100:.2f}%")


def main():
    """
    Example evaluation script.
    """
    import sys
    sys.path.append('..')
    
    from data.dataset import create_dataloader
    from torchvision import transforms
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mimic_root', type=str, required=True, help='Path to MIMIC-CXR')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20])
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load data
    print(f"Loading {args.split} dataset...")
    dataloader = create_dataloader(
        mimic_root=args.mimic_root,
        split=args.split,
        batch_size=args.batch_size,
        min_images_per_study=2,
        num_workers=4,
        transform=transform,
        return_all_images=True,
        use_flat_collate=True,
        shuffle=False,
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create evaluator
    evaluator = RetrievalEvaluator(
        model=model,
        device=device,
        k_values=args.k_values,
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(dataloader)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    import json
    results_path = args.checkpoint.replace('.pt', '_results.json')
    with open(results_path, 'w') as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()} 
                   for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()