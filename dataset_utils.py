import os
import torch
from datasets import load_dataset
from PIL import Image
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np

class BirdDatasetProcessor:
    def __init__(self, base_dir="datasets/bird_dataset", sample_ratio=1.0, random_seed=42):
        """
        Initialize the dataset processor
        Args:
            base_dir (str): Directory to store the processed dataset
            sample_ratio (float): Ratio of data to use (0.0 to 1.0)
            random_seed (int): Random seed for reproducibility
        """
        self.base_dir = base_dir
        self.dataset = None
        self.yaml_path = os.path.join('datasets', 'dataset.yaml')
        self.sample_ratio = max(0.0, min(1.0, sample_ratio))  # Clamp between 0 and 1
        self.random_seed = random_seed
        
    def download_dataset(self):
        """Download the birdsnap dataset"""
        print("Downloading dataset...")
        # Set HuggingFace cache directory to local datasets folder
        os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'datasets', '.cache')
        self.dataset = load_dataset("sasha/birdsnap")
        return self.dataset
    
    def setup_directories(self):
        """Create necessary directories for dataset"""
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.base_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, split, 'labels'), exist_ok=True)
    
    def create_splits(self, train_ratio=0.7, val_ratio=0.15):
        """Create dataset splits with optional sampling"""
        if self.dataset is None:
            self.download_dataset()
            
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
            
        total_size = len(self.dataset['train'])
        
        # If using sample_ratio < 1, first sample the dataset
        if self.sample_ratio < 1.0:
            sample_size = int(total_size * self.sample_ratio)
            # Randomly sample indices
            all_indices = np.random.permutation(total_size)[:sample_size]
            print(f"Using {sample_size} samples ({self.sample_ratio*100:.1f}% of full dataset)")
        else:
            all_indices = np.random.permutation(total_size)
            
        # Calculate split sizes based on the sampled dataset size
        actual_size = len(all_indices)
        train_size = int(actual_size * train_ratio)
        val_size = int(actual_size * val_ratio)
        
        # Create splits
        splits = {
            'train': all_indices[:train_size].tolist(),
            'val': all_indices[train_size:train_size + val_size].tolist(),
            'test': all_indices[train_size + val_size:].tolist()
        }
        
        # Print split sizes
        print("\nDataset split sizes:")
        for split_name, indices in splits.items():
            print(f"{split_name}: {len(indices)} images")
            
        return splits
    
    def create_yaml(self):
        """Create dataset.yaml file with dataset information"""
        if self.dataset is None:
            self.download_dataset()
            
        classes = self.dataset['train'].features['label'].names
        yaml_content = {
            'train': os.path.join(os.getcwd(), self.base_dir, 'train', 'images'),
            'val': os.path.join(os.getcwd(), self.base_dir, 'val', 'images'),
            'test': os.path.join(os.getcwd(), self.base_dir, 'test', 'images'),
            'nc': len(classes),
            'names': classes
        }
        
        with open(self.yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        return yaml_content
    
    def process_dataset(self):
        """Process and save the dataset"""
        self.setup_directories()
        splits = self.create_splits()
        yaml_content = self.create_yaml()
        
        # Track class distribution
        class_counts = {split: {} for split in splits.keys()}
        
        for split_name, indices in splits.items():
            print(f"\nProcessing {split_name} split...")
            for idx in tqdm(indices):
                sample = self.dataset['train'][idx]
                
                # Save image
                image = sample['image']
                image_path = os.path.join(self.base_dir, split_name, 'images', f"{idx}.jpg")
                image.save(image_path)
                
                # Save label
                label = sample['label']
                label_path = os.path.join(self.base_dir, split_name, 'labels', f"{idx}.txt")
                
                # Update class distribution
                class_counts[split_name][label] = class_counts[split_name].get(label, 0) + 1
                
                # Save label as a simple number (class index)
                with open(label_path, 'w') as f:
                    f.write(f"{label}\n")
        
        # Print class distribution summary
        print("\nClass distribution summary:")
        for split_name, counts in class_counts.items():
            n_classes = len(counts)
            print(f"\n{split_name}:")
            print(f"Number of classes: {n_classes}")
            print(f"Average samples per class: {np.mean(list(counts.values())):.1f}")
            print(f"Min samples per class: {min(counts.values())}")
            print(f"Max samples per class: {max(counts.values())}")
        
        return yaml_content

    def get_dataset_info(self):
        """Get information about the processed dataset"""
        if not os.path.exists(self.yaml_path):
            return None
        
        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)

def prepare_dataset(base_dir="datasets/bird_dataset", sample_ratio=1.0, random_seed=42):
    """Helper function to prepare the dataset
    Args:
        base_dir (str): Directory to store the processed dataset
        sample_ratio (float): Ratio of data to use (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
    """
    processor = BirdDatasetProcessor(base_dir, sample_ratio, random_seed)
    return processor.process_dataset()

def get_dataset_labels(dataset_name):
    """Get class labels for a specific dataset
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'birdsnap', etc.)
    Returns:
        list: List of class names
    """
    # Check if there's a yaml file for the dataset
    yaml_path = os.path.join('datasets', f'{dataset_name}.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
            if dataset_info and 'names' in dataset_info:
                return dataset_info['names']
    
    # Fallback to default class names for known datasets
    if dataset_name == 'cifar10':
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name == 'birdsnap':
        # Try loading from the dataset directly
        try:
            dataset = load_dataset("sasha/birdsnap")
            return dataset['train'].features['label'].names
        except Exception as e:
            print(f"Error loading BirdSnap class names: {e}")
            return None
    
    return None 