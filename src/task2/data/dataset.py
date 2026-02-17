import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random


class MultimodalRadiographDataset(Dataset):
    """
    Dataset for multimodal radiograph retrieval using image-text contrastive learning.
    
    Directory structure:
        files/
            p10/
                p10000032/
                    s50414267/
                        image1.jpg
                        image2.jpg
                    s50414267.txt
                    s53189527/
                        image1.jpg
                    s53189527.txt
    
    Each study has:
        - Multiple images in study_id/ folder
        - One text report in study_id.txt file
    """
    
    def __init__(
        self,
        mimic_root: str,
        split: str = 'train',
        min_images_per_study: int = 2,
        transform=None,
        return_all_images: bool = True,
        max_images_per_study: Optional[int] = None,
    ):
        """
        Args:
            mimic_root: Root directory of MIMIC-CXR dataset (contains 'files/' folder)
            split: Dataset split ('train', 'val', 'test') - for future use
            min_images_per_study: Minimum number of images a study must have to be included
            transform: Optional transform to apply to images
            return_all_images: If True, return all images per study; if False, sample one
            max_images_per_study: Maximum number of images to return per study (None = no limit)
        """
        self.mimic_root = Path(mimic_root)
        self.files_root = self.mimic_root / 'files'
        self.split = split
        self.min_images_per_study = min_images_per_study
        self.transform = transform
        self.return_all_images = return_all_images
        self.max_images_per_study = max_images_per_study
        
        # Scan directory and build study index
        print(f"Scanning {self.files_root} for studies...")
        self.studies = self._build_study_index()
        
        print(f"Found {len(self.studies)} studies with >= {min_images_per_study} images")
    
    def _build_study_index(self) -> List[Dict]:
        """
        Scan the directory structure and build an index of all studies.
        
        Returns:
            List of study dictionaries with keys:
                - patient_id: e.g., 'p10000032'
                - study_id: e.g., 's50414267'
                - image_paths: List of paths to images
                - report_path: Path to text report
                - patient_folder: e.g., 'p10' (for organizing)
                - num_images: Number of images in this study
        """
        studies = []
        
        # Iterate through patient folders (p10, p11, p12, ...)
        if not self.files_root.exists():
            raise ValueError(f"Files root not found: {self.files_root}")
        
        patient_folders = sorted([d for d in self.files_root.iterdir() if d.is_dir()])
        
        for patient_folder in patient_folders:
            # e.g., patient_folder = files/p10
            patient_folder_name = patient_folder.name  # 'p10'
            
            # Iterate through individual patient directories
            patient_dirs = sorted([d for d in patient_folder.iterdir() if d.is_dir()])
            
            for patient_dir in patient_dirs:
                # e.g., patient_dir = files/p10/p10000032
                patient_id = patient_dir.name  # 'p10000032'
                
                # Find all study directories and report files
                study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
                
                for study_dir in study_dirs:
                    # e.g., study_dir = files/p10/p10000032/s50414267
                    study_id = study_dir.name  # 's50414267'
                    
                    # Get all images in this study
                    image_files = sorted(list(study_dir.glob('*.jpg')))
                    
                    # Get corresponding report file
                    report_path = patient_dir / f"{study_id}.txt"
                    
                    # Only include studies with:
                    # 1. At least min_images_per_study images
                    # 2. A valid report file
                    if len(image_files) >= self.min_images_per_study and report_path.exists():
                        studies.append({
                            'patient_id': patient_id,
                            'study_id': study_id,
                            'image_paths': [str(img) for img in image_files],
                            'report_path': str(report_path),
                            'patient_folder': patient_folder_name,
                            'num_images': len(image_files)
                        })
        
        return studies
    
    def _load_report_text(self, report_path: str) -> str:
        """Load text report from file."""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Error loading report {report_path}: {e}")
            return ""
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        """
        Get a study with its images and report.
        
        Returns:
            Dict with keys:
                - study_id: Study identifier
                - patient_id: Patient identifier
                - images: List of PIL Images (or single image if not return_all_images)
                - image_paths: List of image paths
                - report_text: String containing the radiograph report
                - num_images: Number of images in this study
        """
        study = self.studies[idx]
        
        # Load report text
        report_text = self._load_report_text(study['report_path'])
        
        # Decide which images to load
        image_paths = study['image_paths']
        
        if not self.return_all_images:
            # Sample one random image
            image_paths = [random.choice(image_paths)]
        elif self.max_images_per_study is not None and len(image_paths) > self.max_images_per_study:
            # Sample up to max_images_per_study
            image_paths = random.sample(image_paths, self.max_images_per_study)
        
        # Load images
        images = []
        valid_paths = []
        for img_path in image_paths:
            img = self._load_image(img_path)
            if img is not None:
                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)
                valid_paths.append(img_path)
        
        return {
            'study_id': study['study_id'], # Study ID (e.g., 's50414267')
            'patient_id': study['patient_id'], # Patient ID (e.g., 'p10000032')
            'images': images,  # List of PIL Images (or transformed tensors)
            'image_paths': valid_paths, # List of valid image paths that were loaded
            'report_text': report_text, # Text report for this study
            'num_images': len(images), # Number of valid images loaded
        }


def collate_fn_multimodal(batch):
    """
    Collate function for multimodal retrieval.
    
    Handles variable number of images per study.
    
    Args:
        batch: List of dicts from MultimodalRadiographDataset
    
    Returns:
        Dict with:
            - study_ids: List of study IDs
            - patient_ids: List of patient IDs
            - images: List of lists of images (each inner list = one study's images)
            - image_paths: List of lists of image paths
            - report_texts: List of report text strings
            - num_images: List of number of images per study
    """
    study_ids = [item['study_id'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    images = [item['images'] for item in batch]  # List of lists
    image_paths = [item['image_paths'] for item in batch]
    report_texts = [item['report_text'] for item in batch]
    num_images = [item['num_images'] for item in batch]
    
    return {
        'study_ids': study_ids,
        'patient_ids': patient_ids,
        'images': images,
        'image_paths': image_paths,
        'report_texts': report_texts,
        'num_images': num_images,
    }


def collate_fn_flat(batch):
    """
    Alternative collate function that flattens all images into a single batch.
    
    This is useful if you want to process all images together and keep track
    of which study they belong to.
    
    Args:
        batch: List of dicts from MultimodalRadiographDataset
    
    Returns:
        Dict with:
            - study_ids: List of study IDs (one per image, repeated)
            - images: Flattened list of all images
            - image_paths: Flattened list of all image paths
            - report_texts: List of report texts (one per image, repeated)
            - study_indices: List indicating which study each image belongs to
            - images_per_study: List of number of images per study
    """
    all_images = []
    all_image_paths = []
    all_study_ids = []
    all_report_texts = []
    study_indices = []
    images_per_study = []
    
    for batch_idx, item in enumerate(batch):
        num_imgs = item['num_images']
        images_per_study.append(num_imgs)
        
        for img, img_path in zip(item['images'], item['image_paths']):
            all_images.append(img)
            all_image_paths.append(img_path)
            all_study_ids.append(item['study_id'])
            all_report_texts.append(item['report_text'])
            study_indices.append(batch_idx)
    
    return {
        'study_ids': all_study_ids,
        'images': all_images,
        'image_paths': all_image_paths,
        'report_texts': all_report_texts,
        'study_indices': torch.tensor(study_indices, dtype=torch.long),
        'images_per_study': images_per_study,
    }


def create_dataloader(
    mimic_root: str,
    split: str = 'train',
    batch_size: int = 16,
    min_images_per_study: int = 2,
    num_workers: int = 4,
    transform=None,
    return_all_images: bool = True,
    max_images_per_study: Optional[int] = None,
    use_flat_collate: bool = False,
    shuffle: bool = None,
):
    """
    Create a DataLoader for multimodal radiograph retrieval.
    
    Args:
        mimic_root: Root directory of MIMIC-CXR dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        min_images_per_study: Minimum images per study to include
        num_workers: Number of data loading workers
        transform: Image transforms
        return_all_images: Whether to return all images per study
        max_images_per_study: Maximum images to return per study
        use_flat_collate: Use flattened collate function
        shuffle: Whether to shuffle (defaults to True for train, False otherwise)
    
    Returns:
        DataLoader
    """
    dataset = MultimodalRadiographDataset(
        mimic_root=mimic_root,
        split=split,
        min_images_per_study=min_images_per_study,
        transform=transform,
        return_all_images=return_all_images,
        max_images_per_study=max_images_per_study,
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    collate_fn = collate_fn_flat if use_flat_collate else collate_fn_multimodal
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


def main_example():
    """Example usage of the dataset."""
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataloader
    dataloader = create_dataloader(
        mimic_root="/datasets/MIMIC-CXR", 
        split='train',
        batch_size=4,
        min_images_per_study=2,  # Only studies with 2+ images
        num_workers=4,
        transform=transform,
        return_all_images=True,  # Return all images per study
        max_images_per_study=None,  # No limit
        use_flat_collate=False,  # Use nested structure
    )
    
    # Iterate over batches
    print("\nExample batch (nested structure):")
    for batch in dataloader:
        print(f"Batch size: {len(batch['study_ids'])}")
        print(f"Study IDs: {batch['study_ids']}")
        print(f"Images per study: {batch['num_images']}")
        
        # Access first study's images
        first_study_images = batch['images'][0]
        print(f"First study has {len(first_study_images)} images")
        if len(first_study_images) > 0:
            print(f"First image shape: {first_study_images[0].shape}")
        
        # Access first study's report
        print(f"First report (first 100 chars): {batch['report_texts'][0][:100]}...")
        break
    
    # Example with flat collate
    print("\n\nExample batch (flat structure):")
    dataloader_flat = create_dataloader(
        mimic_root="/datasets/MIMIC-CXR",
        split='train',
        batch_size=4,
        min_images_per_study=2,
        num_workers=4,
        transform=transform,
        return_all_images=True,
        use_flat_collate=True,  # Flatten all images
    )
    
    for batch in dataloader_flat:
        print(f"Total images in batch: {len(batch['images'])}")
        print(f"Study indices: {batch['study_indices']}")
        print(f"Images per study: {batch['images_per_study']}")
        
        # All images are in a flat list
        if len(batch['images']) > 0:
            print(f"First image shape: {batch['images'][0].shape}")
        break


if __name__ == "__main__":
    main_example()