
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

            
    Supports two split strategies:
    1. Study-level: Each study independently assigned to train/val/test
    2. Patient-level: All studies from same patient stay in same split
    """
    
    def __init__(
        self,
        mimic_root: str,
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_level: str = 'study',  # 'study' or 'patient'
        min_images_per_study: int = 2,
        transform=None,
        return_all_images: bool = True,
        max_images_per_study: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            mimic_root: Root directory of MIMIC-CXR dataset
            split: Dataset split ('train', 'val', 'test')
            split_ratios: (train, val, test) ratios, must sum to 1.0
            split_level: 'study' or 'patient'
                - 'study': Each study independently split
                - 'patient': All studies from same patient stay together
            min_images_per_study: Minimum number of images a study must have
            transform: Optional transform to apply to images
            return_all_images: If True, return all images per study
            max_images_per_study: Maximum images to return per study
            seed: Random seed for reproducible splits
        """
        self.mimic_root = Path(mimic_root)
        self.files_root = self.mimic_root / 'files'
        self.split = split
        self.split_ratios = split_ratios
        self.split_level = split_level
        self.min_images_per_study = min_images_per_study
        self.transform = transform
        self.return_all_images = return_all_images
        self.max_images_per_study = max_images_per_study
        self.seed = seed
        
        # Validate inputs
        assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(split_ratios)}"
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert split_level in ['study', 'patient'], f"Split level must be 'study' or 'patient', got {split_level}"
        
        # Scan directory and build study index
        print(f"\nScanning {self.files_root} for studies...")
        all_studies = self._build_study_index()
        
        # Split into train/val/test
        print(f"Splitting data at {split_level} level with ratios {split_ratios}...")
        self.studies = self._split_studies(all_studies)
        
        print(f"Split: {split}")
        print(f"  Total studies: {len(self.studies)}")
        if split_level == 'patient':
            unique_patients = len(set(s['patient_id'] for s in self.studies))
            print(f"  Unique patients: {unique_patients}")
        print()
    
    def _build_study_index(self) -> List[Dict]:
        """Build index of all studies (before splitting)."""
        studies = []
        
        if not self.files_root.exists():
            raise ValueError(f"Files root not found: {self.files_root}")
        
        patient_folders = sorted([d for d in self.files_root.iterdir() if d.is_dir()])
        
        for patient_folder in patient_folders:
            patient_folder_name = patient_folder.name  # 'p10'
            patient_dirs = sorted([d for d in patient_folder.iterdir() if d.is_dir()])
            
            for patient_dir in patient_dirs:
                patient_id = patient_dir.name  # 'p10000032'
                study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
                
                for study_dir in study_dirs:
                    study_id = study_dir.name  # 's50414267'
                    image_files = sorted(list(study_dir.glob('*.jpg')))
                    report_path = patient_dir / f"{study_id}.txt"
                    
                    if len(image_files) >= self.min_images_per_study and report_path.exists():
                        studies.append({
                            'patient_id': patient_id,
                            'study_id': study_id,
                            'image_paths': [str(img) for img in image_files],
                            'report_path': str(report_path),
                            'patient_folder': patient_folder_name,
                            'num_images': len(image_files)
                        })
        
        print(f"Found {len(studies)} total studies with >= {self.min_images_per_study} images")
        return studies
    
    def _split_studies(self, all_studies: List[Dict]) -> List[Dict]:
        """
        Split studies into train/val/test.
        
        Simple approach:
        1. Sort all items
        2. Shuffle with seed for reproducibility
        3. Split into train/val/test using simple slicing
        """
        if self.split_level == 'study':
            return self._split_by_study(all_studies)
        else:
            return self._split_by_patient(all_studies)
    
    def _split_by_study(self, all_studies: List[Dict]) -> List[Dict]:
        """
        Study-level split: Each study independently assigned.
        
        Simple approach:
        1. Sort studies by study_id for consistency
        2. Shuffle with seed
        3. Split using ratios
        """
        # Sort for consistency
        sorted_studies = sorted(all_studies, key=lambda x: x['study_id'])
        
        # Shuffle with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(sorted_studies)
        
        # Calculate split indices
        total = len(sorted_studies)
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split the list
        if self.split == 'train':
            return sorted_studies[:train_end]
        elif self.split == 'val':
            return sorted_studies[train_end:val_end]
        else:  # test
            return sorted_studies[val_end:]
    
    def _split_by_patient(self, all_studies: List[Dict]) -> List[Dict]:
        """
        Patient-level split: All studies from same patient stay together.
        
        Simple approach:
        1. Group studies by patient
        2. Sort patients for consistency
        3. Shuffle patients with seed
        4. Split patients into train/val/test
        5. Collect all studies from patients in this split
        """
        # Group studies by patient
        patient_to_studies = defaultdict(list)
        for study in all_studies:
            patient_to_studies[study['patient_id']].append(study)
        
        # Get sorted list of patient IDs
        patient_ids = sorted(patient_to_studies.keys())
        
        # Shuffle patients with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(patient_ids)
        
        # Calculate split indices for patients
        total_patients = len(patient_ids)
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        train_end = int(total_patients * train_ratio)
        val_end = train_end + int(total_patients * val_ratio)
        
        # Split patients
        if self.split == 'train':
            split_patients = patient_ids[:train_end]
        elif self.split == 'val':
            split_patients = patient_ids[train_end:val_end]
        else:  # test
            split_patients = patient_ids[val_end:]
        
        # Collect all studies from these patients
        studies_for_split = []
        for patient_id in split_patients:
            studies_for_split.extend(patient_to_studies[patient_id])
        
        return studies_for_split
    
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
        """Get a study with its images and report."""
        study = self.studies[idx]
        
        # Load report text
        report_text = self._load_report_text(study['report_path'])
        
        # Decide which images to load
        image_paths = study['image_paths']
        
        if not self.return_all_images:
            image_paths = [random.choice(image_paths)]
        elif self.max_images_per_study is not None and len(image_paths) > self.max_images_per_study:
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
            'study_id': study['study_id'],
            'patient_id': study['patient_id'],
            'images': images,
            'image_paths': valid_paths,
            'report_text': report_text,
            'num_images': len(images),
        }


def collate_fn_multimodal(batch):
    """Collate function for multimodal retrieval (nested structure)."""
    study_ids = [item['study_id'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    images = [item['images'] for item in batch]
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
    """Collate function that flattens all images (for contrastive learning)."""
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
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_level: str = 'study',
    batch_size: int = 16,
    min_images_per_study: int = 2,
    num_workers: int = 4,
    transform=None,
    return_all_images: bool = True,
    max_images_per_study: Optional[int] = None,
    use_flat_collate: bool = False,
    shuffle: bool = None,
    seed: int = 42,
):
    """
    Create a DataLoader for multimodal radiograph retrieval.
    
    Args:
        mimic_root: Root directory of MIMIC-CXR dataset
        split: Dataset split ('train', 'val', 'test')
        split_ratios: (train, val, test) ratios, must sum to 1.0
        split_level: 'study' or 'patient'
            - 'study': Each study independently split
            - 'patient': All studies from same patient stay together
        batch_size: Batch size
        min_images_per_study: Minimum images per study to include
        num_workers: Number of data loading workers
        transform: Image transforms
        return_all_images: Whether to return all images per study
        max_images_per_study: Maximum images to return per study
        use_flat_collate: Use flattened collate function
        shuffle: Whether to shuffle (defaults to True for train, False otherwise)
        seed: Random seed for reproducible splits
    
    Returns:
        DataLoader
    """
    dataset = MultimodalRadiographDataset(
        mimic_root=mimic_root,
        split=split,
        split_ratios=split_ratios,
        split_level=split_level,
        min_images_per_study=min_images_per_study,
        transform=transform,
        return_all_images=return_all_images,
        max_images_per_study=max_images_per_study,
        seed=seed,
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


def create_all_dataloaders(
    mimic_root: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_level: str = 'study',
    batch_size: int = 16,
    min_images_per_study: int = 2,
    num_workers: int = 4,
    transform=None,
    use_flat_collate: bool = False,
    seed: int = 42,
):
    """
    Create train, val, and test dataloaders with the same split configuration.
    
    Args:
        mimic_root: Root directory of MIMIC-CXR dataset
        split_ratios: (train, val, test) ratios
        split_level: 'study' or 'patient'
        batch_size: Batch size
        min_images_per_study: Minimum images per study
        num_workers: Number of workers
        transform: Image transforms
        use_flat_collate: Use flat collate function
        seed: Random seed for splits
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = create_dataloader(
        mimic_root=mimic_root,
        split='train',
        split_ratios=split_ratios,
        split_level=split_level,
        batch_size=batch_size,
        min_images_per_study=min_images_per_study,
        num_workers=num_workers,
        transform=transform,
        use_flat_collate=use_flat_collate,
        shuffle=True,
        seed=seed,
    )
    
    val_loader = create_dataloader(
        mimic_root=mimic_root,
        split='val',
        split_ratios=split_ratios,
        split_level=split_level,
        batch_size=batch_size,
        min_images_per_study=min_images_per_study,
        num_workers=num_workers,
        transform=transform,
        use_flat_collate=use_flat_collate,
        shuffle=False,
        seed=seed,
    )
    
    test_loader = create_dataloader(
        mimic_root=mimic_root,
        split='test',
        split_ratios=split_ratios,
        split_level=split_level,
        batch_size=batch_size,
        min_images_per_study=min_images_per_study,
        num_workers=num_workers,
        transform=transform,
        use_flat_collate=use_flat_collate,
        shuffle=False,
        seed=seed,
    )
    
    return train_loader, val_loader, test_loader


def main_example():
    """Example usage with simple splitting."""
    print("="*70)
    print("Example 1: Study-Level Split (70/15/15)")
    print("="*70)
    print("\nHow it works:")
    print("1. Collect all studies")
    print("2. Sort by study_id")
    print("3. Shuffle with seed=42")
    print("4. Split: first 70% → train, next 15% → val, last 15% → test")
    
    train_loader, val_loader, test_loader = create_all_dataloaders(
        mimic_root="/datasets/MIMIC-CXR",
        split_ratios=(0.7, 0.15, 0.15),
        split_level='study',
        batch_size=4,
        min_images_per_study=2,
        use_flat_collate=True,
        seed=42,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} studies")
    print(f"  Val:   {len(val_loader.dataset)} studies")
    print(f"  Test:  {len(test_loader.dataset)} studies")
    
    print("\n" + "="*70)
    print("Example 2: Patient-Level Split (80/10/10)")
    print("="*70)
    print("\nHow it works:")
    print("1. Group studies by patient")
    print("2. Sort patients by patient_id")
    print("3. Shuffle patients with seed=42")
    print("4. Split patients: first 80% → train, next 10% → val, last 10% → test")
    print("5. Collect all studies from each patient group")
    
    train_loader, val_loader, test_loader = create_all_dataloaders(
        mimic_root="/datasets/MIMIC-CXR",
        split_ratios=(0.8, 0.1, 0.1),
        split_level='patient',
        batch_size=4,
        min_images_per_study=2,
        use_flat_collate=True,
        seed=42,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} studies")
    print(f"  Val:   {len(val_loader.dataset)} studies")
    print(f"  Test:  {len(test_loader.dataset)} studies")


if __name__ == "__main__":
    main_example()