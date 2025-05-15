import os
import torch
from PIL import Image
import yaml
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    def __init__(self, base_dir="datasets", sample_ratio=1.0, random_seed=42):
        """
        Initialize the base preprocessor
        Args:
            base_dir (str): Base directory for storing datasets
            sample_ratio (float): Ratio of data to use (0.0 to 1.0)
            random_seed (int): Random seed for reproducibility
        """
        self.base_dir = base_dir
        self.sample_ratio = max(0.0, min(1.0, sample_ratio))
        self.random_seed = random_seed
        
    def setup_directories(self, dataset_name):
        """Create necessary directories for dataset"""
        dataset_dir = os.path.join(self.base_dir, dataset_name)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)
        return dataset_dir
    
    def create_splits(self, total_size, train_ratio=0.7, val_ratio=0.15):
        """Create dataset splits"""
        # Convert indices to python list
        all_indices = list(range(total_size))
        np.random.shuffle(all_indices)
        
        if self.sample_ratio < 1.0:
            sample_size = int(total_size * self.sample_ratio)
            all_indices = all_indices[:sample_size]
            
        train_size = int(len(all_indices) * train_ratio)
        val_size = int(len(all_indices) * val_ratio)
        
        return {
            'train': all_indices[:train_size],
            'val': all_indices[train_size:train_size + val_size],
            'test': all_indices[train_size + val_size:]
        }
    
    def save_metadata(self, dataset_dir, metadata):
        """Save dataset metadata to yaml file"""
        yaml_path = os.path.join(dataset_dir, 'metadata.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(metadata, f)
    
    @abstractmethod
    def process(self):
        """Process and save the dataset"""
        pass

def load_species_info():
    """Load bird species information from metainfo file"""
    species_file = "dataset_metainfo/species.txt"
    if not os.path.exists(species_file):
        raise ValueError(f"Species info file not found: {species_file}")
        
    species_info = {}
    with open(species_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            if not line.strip():  # Skip empty lines
                continue
                
            # Parse line with all fields concatenated (no proper tab separation)
            if '\t' not in line:
                content = line.strip()
                # Find first dot after ID
                dot_pos = content.find('.')
                if dot_pos == -1:
                    continue
                    
                # Extract parts
                idx = int(content[:dot_pos])
                rest = content[dot_pos+1:].strip()
                
                # Last word is the dir/label name
                parts = rest.split()
                label = parts[-1].lower()  # Convert to lowercase for consistency
                common_name = ' '.join(parts[:-1])
                
            else:
                # Parse tab-separated fields
                parts = line.strip().split('\t')
                if len(parts) < 4:  # Need at least ID, common name, scientific name, and dir
                    continue
                    
                idx = int(parts[0])
                label = parts[3].lower()  # Last column is the dir/label name
                common_name = parts[1]  # Second column is common name
                
            # Convert to 0-based index and store
            species_info[label] = {
                'id': idx - 1,
                'common_name': common_name
            }
    
    return species_info
    
class BirdDatasetPreprocessor(BasePreprocessor):
    def __init__(self, base_dir="datasets", sample_ratio=1.0, random_seed=42):
        super().__init__(base_dir=base_dir, sample_ratio=sample_ratio, random_seed=random_seed)
        # Set dataset name
        self.base_dir = os.path.join(base_dir, "birdsnap")
        # Load predefined species info
        self.species_info = load_species_info()
        
    def process(self):
        """Process the birdsnap dataset"""
        from datasets import load_dataset
        
        print("Processing BirdSnap dataset...")
        dataset = load_dataset("sasha/birdsnap")
        
        # Create label mappings from predefined species info
        # First create label_to_idx mapping
        self.label_to_idx = {name: info['id'] for name, info in self.species_info.items()}
        # Then create reverse mapping
        self.idx_to_label = {id_: name for name, id_ in self.label_to_idx.items()}
        
        # Setup output directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.base_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, split, 'labels'), exist_ok=True)
        splits = self.create_splits(len(dataset['train']))
        
        # Setup progress tracking
        total_samples = sum(len(indices) for indices in splits.values())
        overall_progress = tqdm(total=total_samples, desc="Overall", unit='img', position=0)
        
        # Process each split
        class_counts = {split: {} for split in splits.keys()}
        for split_name, indices in splits.items():
            print(f"\nProcessing {split_name} split ({len(indices)} samples)...")
            progress_bar = tqdm(indices, desc=f"{split_name}", unit='img', position=1, leave=False)
            for idx in progress_bar:
                try:
                    sample = dataset['train'][idx]
                    
                    # Handle image
                    image = sample['image']
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image_path = os.path.join(self.base_dir, split_name, 'images', f"{idx}.jpg")
                    image.save(image_path)
                    
                    # Save label using predefined mapping
                    # Convert dataset label to lowercase to match our mapping
                    label_name = sample['label'].lower()
                    if label_name not in self.label_to_idx:
                        print(f"Warning: Label '{sample['label']}' not found in species info")
                        continue
                except (OSError, IOError, IndexError) as e:
                    # Handle corrupted image files or other errors
                    print(f"\nError processing sample {idx}: {str(e)}")
                    overall_progress.update(1)  # Update progress bar even for skipped samples
                    continue
                    
                label_idx = self.label_to_idx[label_name]
                common_name = self.species_info[label_name]['common_name']
                # Use original capitalization for display
                display_name = sample['label']
                progress_bar.set_postfix({'class': f"{display_name} ({label_idx}) - {common_name}"})
                overall_progress.update(1)
                
                label_path = os.path.join(self.base_dir, split_name, 'labels', f"{idx}.txt")
                with open(label_path, 'w') as f:
                    f.write(f"{label_idx}\n")
                
                class_counts[split_name][label_idx] = class_counts[split_name].get(label_idx, 0) + 1
        
        # Save metadata
        metadata = {
            'name': 'birdsnap',
            'num_classes': len(self.species_info),
            'classes': self.idx_to_label,
            'label_mapping': self.label_to_idx,
            'splits': {
                split: {
                    'size': len(indices),
                    'class_distribution': class_counts[split]
                }
                for split, indices in splits.items()
            }
        }
        self.save_metadata(self.base_dir, metadata)
        
        # Print summary
        print("\nDataset summary:")
        print(f"Total classes: {len(self.species_info)}")
        for split_name, split_info in metadata['splits'].items():
            print(f"\n{split_name}:")
            print(f"Samples: {split_info['size']}")
            dist = np.array(list(split_info['class_distribution'].values()))
            if len(dist) > 0:
                print(f"Samples per class: {dist.mean():.1f} ± {dist.std():.1f}")
                print(f"Min samples per class: {min(dist)}")
                print(f"Max samples per class: {max(dist)}")
        
        return metadata

# Dataset registry
DATASET_REGISTRY = {
    'birdsnap': BirdDatasetPreprocessor
}

def get_dataset_metadata(dataset_name, base_dir="datasets"):
    """Get dataset metadata if it exists"""
    yaml_path = os.path.join(base_dir, dataset_name, 'metadata.yaml')
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_dataset(dataset_name, base_dir="datasets", sample_ratio=1.0, random_seed=42):
    """Preprocess a dataset to the unified format
    
    Args:
        dataset_name (str): Name of the dataset
        base_dir (str): Base directory for storing datasets
        sample_ratio (float): Ratio of data to use
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dataset metadata
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(DATASET_REGISTRY.keys())}")
    
    # Create processor instance
    processor = DATASET_REGISTRY[dataset_name](
        base_dir=base_dir,
        sample_ratio=sample_ratio,
        random_seed=random_seed
    )
    
    # Process dataset and return metadata
    return processor.process()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess bird classification datasets into unified format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset', choices=list(DATASET_REGISTRY.keys()),
                      help='Dataset to preprocess')
    parser.add_argument('--base-dir', default='datasets',
                      help='Base directory for storing datasets')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                      help='Ratio of data to use for faster development/testing')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true',
                      help='Force reprocessing even if dataset exists')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check if dataset already exists
    metadata = get_dataset_metadata(args.dataset, args.base_dir)
    if metadata is not None and not args.force:
        print(f"\nDataset {args.dataset} already exists in {args.base_dir}")
        print("Use --force to reprocess")
        print("\nExisting dataset info:")
        print(f"Number of classes: {metadata['num_classes']}")
        for split, info in metadata['splits'].items():
            print(f"{split}: {info['size']} samples")
    else:
        # Process dataset
        metadata = preprocess_dataset(
            args.dataset,
            base_dir=args.base_dir,
            sample_ratio=args.sample_ratio,
            random_seed=args.seed
        )
