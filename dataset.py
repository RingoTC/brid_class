import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from preprocess import get_dataset_metadata
from sklearn.model_selection import KFold

class ImageClassificationDataset(Dataset):
    """Dataset for image classification with support for k-fold cross validation"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (integers)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self._transform = None
        self.transform = transform
        
    @property
    def transform(self):
        """Get the current transform"""
        return self._transform
        
    @transform.setter
    def transform(self, transform):
        """Set the transform and handle ViT processor case"""
        self._transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Handle ViT processor case
        if hasattr(self.transform, '__class__') and 'ViTImageProcessor' in self.transform.__class__.__name__:
            inputs = self.transform(images=image, return_tensors="pt")
            # Remove batch dimension
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            inputs['labels'] = label
            return inputs
        
        # Handle regular transforms
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

def load_dataset(dataset_name, base_dir="datasets", transform=None, kfold_config=None):
    """Load a preprocessed dataset
    
    Args:
        dataset_name (str): Name of the dataset to load
        base_dir (str): Base directory containing datasets
        transform: Optional transform to be applied on images
        kfold_config (dict): K-fold configuration with keys:
            - enabled (bool): Whether to use k-fold
            - k (int): Number of folds
            - fold (int): Current fold to use (0-based)
    
    Returns:
        dict: Dataset info containing:
            - train_dataset: Training dataset
            - val_dataset: Validation dataset
            - test_dataset: Test dataset
            - num_classes: Number of classes
            - class_names: Dictionary mapping class indices to names
        or
        list[dict]: List of dataset info dictionaries when using k-fold
    """
    # Get dataset metadata
    metadata = get_dataset_metadata(dataset_name, base_dir)
    if metadata is None:
        raise ValueError(f"Dataset {dataset_name} not found in {base_dir}. Run preprocess.py first.")
    
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    def load_split(split_name):
        """Helper to load a single split"""
        split_dir = os.path.join(dataset_dir, split_name)
        
        # Get sorted image paths
        image_paths = sorted([
            os.path.join(split_dir, 'images', f)
            for f in os.listdir(os.path.join(split_dir, 'images'))
            if f.endswith('.jpg')
        ])
        
        # Load labels
        labels = []
        for img_path in image_paths:
            idx = int(os.path.basename(img_path).split('.')[0])
            label_path = os.path.join(split_dir, 'labels', f"{idx}.txt")
            with open(label_path, 'r') as f:
                labels.append(int(f.read().strip()))
                
        return image_paths, labels
    
    # Load all splits
    splits = {}
    for split in ['train', 'val', 'test']:
        splits[split] = load_split(split)
    
    # Create datasets
    if kfold_config and kfold_config.get('enabled', False):
        k = kfold_config.get('k', 5)
        current_fold = kfold_config.get('fold', 0)
        
        # Combine train and val data
        all_image_paths = splits['train'][0] + splits['val'][0]
        all_labels = splits['train'][1] + splits['val'][1]
        
        # Generate folds
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        folds = list(kf.split(all_image_paths))
        
        # Get current fold indices
        train_idx, val_idx = folds[current_fold]
        
        # Create datasets for current fold
        train_dataset = ImageClassificationDataset(
            [all_image_paths[i] for i in train_idx],
            [all_labels[i] for i in train_idx],
            transform=transform
        )
        val_dataset = ImageClassificationDataset(
            [all_image_paths[i] for i in val_idx],
            [all_labels[i] for i in val_idx],
            transform=transform
        )
    else:
        # Create regular datasets
        train_dataset = ImageClassificationDataset(
            splits['train'][0], splits['train'][1], transform=transform
        )
        val_dataset = ImageClassificationDataset(
            splits['val'][0], splits['val'][1], transform=transform
        )
    
    test_dataset = ImageClassificationDataset(
        splits['test'][0], splits['test'][1], transform=transform
    )
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'num_classes': metadata['num_classes'],
        'class_names': metadata['classes']
    }
