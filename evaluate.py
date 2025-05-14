import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_device
import pandas as pd
from datetime import datetime
from PIL import Image
import yaml
import argparse

def load_model(model_path):
    """Load a fine-tuned ViT model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    return model, processor

def get_dataset_info(dataset_name):
    """Get dataset information from yaml file"""
    yaml_path = os.path.join('datasets', dataset_name, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        raise ValueError(f"Dataset configuration not found at {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, processor, dataset_name, split='test'):
    """Evaluate the model on the specified split"""
    # Load dataset info
    dataset_info = get_dataset_info(dataset_name)
    
    if dataset_info is None:
        raise ValueError("Dataset information not found. Please run training first.")
    
    # Configure device
    device = get_device()
    model.to(device)
    model.eval()
    
    # Get test images directory
    test_dir = dataset_info[split]
    class_names = dataset_info['names']
    n_classes = len(class_names)
    
    # Prepare for evaluation
    all_predictions = []
    all_targets = []
    all_pred_probs = []
    
    # Get all test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process images in batches
    batch_size = 32
    
    print(f"\nEvaluating model on {dataset_name} {split} set...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_images), batch_size)):
            batch_images = test_images[i:i + batch_size]
            batch_inputs = []
            batch_labels = []
            
            for img_name in batch_images:
                # Load and preprocess image
                image_path = os.path.join(test_dir, img_name)
                image = Image.open(image_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                batch_inputs.append({k: v.to(device) for k, v in inputs.items()})
                
                # Get true label
                label_path = os.path.join(os.path.dirname(test_dir), 'labels',
                                        os.path.splitext(img_name)[0] + '.txt')
                with open(label_path, 'r') as f:
                    true_label = int(f.read().split()[0])
                batch_labels.append(true_label)
            
            # Stack batch inputs
            batch_input_dict = {
                k: torch.cat([inputs[k] for inputs in batch_inputs]) 
                for k in batch_inputs[0].keys()
            }
            
            # Get predictions
            outputs = model(**batch_input_dict)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predictions and probabilities
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels)
            all_pred_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_pred_probs = np.array(all_pred_probs)
    
    # Calculate metrics
    metrics = {}
    
    # Overall Accuracy
    metrics['accuracy'] = np.mean(all_predictions == all_targets)
    
    # Per-class F1 scores
    f1_per_class = f1_score(all_targets, all_predictions, average=None)
    metrics['f1_macro'] = np.mean(f1_per_class)
    metrics['f1_per_class'] = {class_names[i]: f1_per_class[i] for i in range(n_classes)}
    
    # ROC-AUC score (one-vs-rest)
    try:
        metrics['roc_auc'] = roc_auc_score(
            np.eye(n_classes)[all_targets],
            all_pred_probs,
            multi_class='ovr',
            average='macro'
        )
    except ValueError as e:
        print(f"Warning: Could not calculate ROC-AUC score: {e}")
        metrics['roc_auc'] = None
    
    # Save detailed classification report
    report = classification_report(all_targets, all_predictions,
                                target_names=class_names,
                                output_dict=True)
    metrics['classification_report'] = report
    
    # Print results
    print(f"\nModel Evaluation Results on {dataset_name}:")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join('evaluation_results', dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_prefix = f"{results_dir}/{split}_{timestamp}"
    
    # Save confusion matrix
    plt.savefig(f"{results_prefix}_confusion_matrix.png")
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro F1-Score', 'ROC-AUC'],
        'Value': [
            metrics['accuracy'],
            metrics['f1_macro'],
            metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'
        ]
    })
    metrics_df.to_csv(f"{results_prefix}_metrics.csv", index=False)
    
    # Save per-class metrics
    per_class_metrics = pd.DataFrame({
        'Class': class_names,
        'F1-Score': [metrics['f1_per_class'][name] for name in class_names],
        'Precision': [report[name]['precision'] for name in class_names],
        'Recall': [report[name]['recall'] for name in class_names]
    })
    per_class_metrics.to_csv(f"{results_prefix}_per_class_metrics.csv", index=False)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate image classification model')
    parser.add_argument('--dataset', type=str, default='birdsnap',
                      help='Dataset to evaluate on (birdsnap, cifar10, cifar100)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to the fine-tuned model')
    parser.add_argument('--split', type=str, default='test',
                      help='Dataset split to evaluate on (train, val, test)')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = f"models/{args.dataset}/best"
    
    # Load model
    model, processor = load_model(args.model_path)
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        processor=processor,
        dataset_name=args.dataset,
        split=args.split
    )

if __name__ == "__main__":
    main() 