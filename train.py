import os
import torch
from datasets import load_dataset
from PIL import Image
import yaml
from tqdm import tqdm
import shutil
from pathlib import Path
from dataset_utils import prepare_dataset
from utils import get_device
import argparse
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import numpy as np

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs['labels'] = self.labels[idx]
        return inputs

def setup_dataset(dataset_name, base_dir="datasets", train_ratio=0.7, val_ratio=0.15, sample_ratio=1.0):
    """Setup dataset and return paths and labels for each split"""
    dataset_dir = os.path.join(base_dir, dataset_name)
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    
    # If dataset already exists, load it directly
    if os.path.exists(yaml_path):
        print(f"Found existing {dataset_name} dataset, loading directly...")
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        split_data = {}
        for split_name in ['train', 'val', 'test']:
            split_dir = os.path.join(dataset_dir, split_name)
            image_paths = sorted([
                os.path.join(split_dir, 'images', f) 
                for f in os.listdir(os.path.join(split_dir, 'images'))
                if f.endswith('.jpg')
            ])
            
            # Apply sample ratio to each split
            if sample_ratio < 1.0:
                num_samples = int(len(image_paths) * sample_ratio)
                image_paths = image_paths[:num_samples]
            
            labels = []
            for img_path in image_paths:
                idx = os.path.basename(img_path).split('.')[0]
                label_path = os.path.join(split_dir, 'labels', f"{idx}.txt")
                with open(label_path, 'r') as f:
                    labels.append(int(f.read().strip()))
            
            split_data[split_name] = {
                'image_paths': image_paths,
                'labels': labels
            }
        
        return yaml_content, split_data

    # Dataset specific configurations
    dataset_configs = {
        'birdsnap': {
            'path': "sasha/birdsnap",
            'split_key': 'train',
            'image_key': 'image',
            'label_key': 'label'
        },
        'cifar10': {
            'path': "cifar10",
            'split_key': 'train',
            'image_key': 'img',
            'label_key': 'label'
        },
        'cifar100': {
            'path': "cifar100",
            'split_key': 'train',
            'image_key': 'img',
            'label_key': 'fine_label'
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(config['path'])
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

    # Process the dataset
    total_size = len(dataset[config['split_key']])
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Create splits
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Save dataset info
    classes = dataset[config['split_key']].features[config['label_key']].names
    yaml_content = {
        'dataset_name': dataset_name,
        'train': os.path.join(os.getcwd(), dataset_dir, 'train', 'images'),
        'val': os.path.join(os.getcwd(), dataset_dir, 'val', 'images'),
        'test': os.path.join(os.getcwd(), dataset_dir, 'test', 'images'),
        'nc': len(classes),
        'names': classes
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    # Process and save images
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    split_data = {}
    for split_name, indices in splits.items():
        print(f"Processing {split_name} split...")
        image_paths = []
        labels = []
        
        for idx in tqdm(indices):
            sample = dataset[config['split_key']][idx]
            
            # Save image
            image = sample[config['image_key']]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            image_path = os.path.join(dataset_dir, split_name, 'images', f"{idx}.jpg")
            image.save(image_path)
            
            # Save label
            label = sample[config['label_key']]
            label_path = os.path.join(dataset_dir, split_name, 'labels', f"{idx}.txt")
            
            with open(label_path, 'w') as f:
                f.write(f"{label}\n")
            
            image_paths.append(image_path)
            labels.append(label)
            
        split_data[split_name] = {
            'image_paths': image_paths,
            'labels': labels
        }

    return yaml_content, split_data

def train_model(split_data, yaml_content, model_name="google/vit-base-patch16-224", 
                num_epochs=100, batch_size=32, learning_rate=2e-5, output_dir=None):
    """Train the ViT model"""
    device = get_device()
    
    # Get number of classes from yaml_content
    num_classes = yaml_content['nc']
    dataset_name = yaml_content['dataset_name']
    print(f"\nNumber of classes in {dataset_name} dataset: {num_classes}")
    
    if output_dir is None:
        output_dir = f"models/{dataset_name}"
    
    # Initialize model and processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    train_dataset = ImageClassificationDataset(
        split_data['train']['image_paths'],
        split_data['train']['labels'],
        processor
    )
    val_dataset = ImageClassificationDataset(
        split_data['val']['image_paths'],
        split_data['val']['labels'],
        processor
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        remove_unused_columns=False,
        logging_steps=50,
    )
    
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": (predictions == eval_pred.label_ids).mean()}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the best model
    best_model_path = os.path.join(output_dir, "best")
    trainer.save_model(best_model_path)
    processor.save_pretrained(best_model_path)
    
    return model, processor

def main():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--dataset', type=str, default='birdsnap',
                      help='Dataset to use (birdsnap, cifar10, cifar100)')
    parser.add_argument('--model', type=str, default='google/vit-base-patch16-224',
                      help='Model architecture to use')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                      help='Ratio of dataset to use (0.0 to 1.0)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for saving models')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare dataset
    print(f"Preparing {args.dataset} dataset with sample ratio {args.sample_ratio}...")
    yaml_content, split_data = setup_dataset(args.dataset, sample_ratio=args.sample_ratio)
    
    # Print dataset sizes
    for split in split_data:
        print(f"Number of samples in {split} set: {len(split_data[split]['image_paths'])}")
    
    # Train model
    model, processor = train_model(
        split_data,
        yaml_content,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    print(f"\nTraining completed! Model saved in {args.output_dir or f'models/{args.dataset}'}/best/")

if __name__ == "__main__":
    main() 