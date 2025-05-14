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

class BirdDataset(Dataset):
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

def setup_dataset():
    """Setup dataset and return paths and labels for each split"""
    # Load the birdsnap dataset
    dataset = load_dataset("sasha/birdsnap")
    
    # Create directories
    base_dir = "bird_dataset"
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)

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

    # Save dataset info
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

    split_data = {}
    for split_name, indices in splits.items():
        print(f"Processing {split_name} split...")
        image_paths = []
        labels = []
        
        for idx in tqdm(indices):
            sample = dataset['train'][idx]
            
            # Save image
            image = sample['image']
            image_path = os.path.join(base_dir, split_name, 'images', f"{idx}.jpg")
            image.save(image_path)
            
            image_paths.append(image_path)
            labels.append(sample['label'])
            
        split_data[split_name] = {
            'image_paths': image_paths,
            'labels': labels
        }

    return yaml_content, split_data

def train_model(split_data, yaml_content, num_epochs=100, batch_size=32, learning_rate=2e-5):
    """Train the ViT model"""
    device = get_device()
    
    # Get number of classes from yaml_content
    num_classes = yaml_content['nc']
    print(f"\nNumber of bird species in dataset: {num_classes}")
    
    # Initialize model and processor
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,  # Use actual number of classes from dataset
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    train_dataset = BirdDataset(
        split_data['train']['image_paths'],
        split_data['train']['labels'],
        processor
    )
    val_dataset = BirdDataset(
        split_data['val']['image_paths'],
        split_data['val']['labels'],
        processor
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="models/finetuned",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        remove_unused_columns=False,
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
    trainer.save_model("models/finetuned/best")
    processor.save_pretrained("models/finetuned/best")
    
    return model, processor

def main():
    parser = argparse.ArgumentParser(description='Train bird classification model')
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
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Prepare dataset
    print(f"Preparing dataset with {args.sample_ratio*100:.1f}% of the full dataset...")
    yaml_content, split_data = setup_dataset()
    
    # Train model
    model, processor = train_model(
        split_data,
        yaml_content,  # Pass yaml_content to get number of classes
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\nTraining completed! Model saved in models/finetuned/best/")

if __name__ == "__main__":
    main() 