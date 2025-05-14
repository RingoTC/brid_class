import os
import torch
from datasets import load_dataset
from PIL import Image
import yaml
from ultralytics import YOLO
from tqdm import tqdm
import shutil
from pathlib import Path
from dataset_utils import prepare_dataset
from utils import get_device
import argparse

def setup_dataset():
    # Load the birdsnap dataset
    dataset = load_dataset("sasha/birdsnap")
    
    # Create directories for YOLO format
    base_dir = "bird_dataset"
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Process the dataset
    total_size = len(dataset['train'])
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Create splits
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create dataset.yaml
    classes = dataset['train'].features['label'].names
    yaml_content = {
        'train': os.path.join(os.getcwd(), base_dir, 'train', 'images'),
        'val': os.path.join(os.getcwd(), base_dir, 'val', 'images'),
        'test': os.path.join(os.getcwd(), base_dir, 'test', 'images'),
        'nc': len(classes),
        'names': classes
    }

    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

    # Process and save images
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    for split_name, indices in splits.items():
        print(f"Processing {split_name} split...")
        for idx in tqdm(indices):
            sample = dataset['train'][idx]
            
            # Save image
            image = sample['image']
            image_path = os.path.join(base_dir, split_name, 'images', f"{idx}.jpg")
            image.save(image_path)
            
            # Create YOLO format label
            label = sample['label']
            label_path = os.path.join(base_dir, split_name, 'labels', f"{idx}.txt")
            
            # YOLO format: class_id x_center y_center width height
            # For classification, we'll use the whole image
            with open(label_path, 'w') as f:
                f.write(f"{label} 0.5 0.5 1.0 1.0\n")

    return yaml_content

def train_model(yaml_path='dataset.yaml', epochs=100, img_size=640, batch_size=16):
    """Train the YOLO model"""
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')

    # Configure device
    device = get_device()
    
    # Adjust batch size for CUDA if available
    if device.type == "cuda":
        # Get available GPU memory and calculate max possible batch size
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
        if gpu_memory >= 16:  # For GPUs with 16GB or more
            batch_size = min(32, batch_size * 2)
        elif gpu_memory >= 8:  # For GPUs with 8GB or more
            batch_size = min(24, int(batch_size * 1.5))
    
    print(f"Training with batch size: {batch_size}")

    # Train the model
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=4,
        patience=20,
        save=True,
        project='bird_classification'
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train bird classification model')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                      help='Ratio of dataset to use (0.0 to 1.0)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Image size for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Prepare dataset with sampling
    print(f"Preparing dataset with {args.sample_ratio*100:.1f}% of the full dataset...")
    yaml_content = prepare_dataset(sample_ratio=args.sample_ratio, random_seed=args.seed)
    
    # Train model
    print("\nStarting training...")
    model = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print("\nTraining completed! Model saved in bird_classification/train/weights/")

if __name__ == "__main__":
    main() 