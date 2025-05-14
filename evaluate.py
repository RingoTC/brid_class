import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from dataset_utils import BirdDatasetProcessor
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

def load_model(model_path=None):
    """Load a ViT model
    If model_path is None, loads the base model for zero-shot evaluation
    """
    if model_path is None:
        # Load base model for zero-shot
        model_name = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=500)
        processor = ViTImageProcessor.from_pretrained(model_name)
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)
    
    return model, processor

def evaluate_model(model, processor, dataset_yaml='dataset.yaml', split='test', model_type="fine-tuned"):
    """Evaluate the model on the specified split"""
    # Load dataset info
    data_processor = BirdDatasetProcessor()
    dataset_info = data_processor.get_dataset_info()
    
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
    
    print(f"Evaluating {model_type} model on {split} set...")
    
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
    print(f"\n{model_type} Model Evaluation Results:")
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
    plt.title(f'Confusion Matrix - {model_type} Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_prefix = f"{results_dir}/{model_type}_{split}_{timestamp}"
    
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

def compare_models(fine_tuned_path):
    """Compare zero-shot and fine-tuned model performance"""
    # Load both models
    zero_shot_model, zero_shot_processor = load_model()
    fine_tuned_model, fine_tuned_processor = load_model(fine_tuned_path)
    
    # Evaluate on test set
    print("\nEvaluating Zero-shot Performance...")
    zero_shot_metrics = evaluate_model(
        zero_shot_model, 
        zero_shot_processor,
        split='test',
        model_type="zero-shot"
    )
    
    print("\nEvaluating Fine-tuned Performance...")
    fine_tuned_metrics = evaluate_model(
        fine_tuned_model,
        fine_tuned_processor,
        split='test',
        model_type="fine-tuned"
    )
    
    # Create comparison plots
    metrics_comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro F1-Score', 'ROC-AUC'],
        'Zero-shot': [
            zero_shot_metrics['accuracy'],
            zero_shot_metrics['f1_macro'],
            zero_shot_metrics['roc_auc'] if zero_shot_metrics['roc_auc'] is not None else 0
        ],
        'Fine-tuned': [
            fine_tuned_metrics['accuracy'],
            fine_tuned_metrics['f1_macro'],
            fine_tuned_metrics['roc_auc'] if fine_tuned_metrics['roc_auc'] is not None else 0
        ]
    })
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    metrics_comparison.plot(x='Metric', y=['Zero-shot', 'Fine-tuned'], kind='bar')
    plt.title('Zero-shot vs Fine-tuned Performance')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evaluation_results/model_comparison.png')
    plt.close()
    
    # Save comparison to CSV
    metrics_comparison.to_csv('evaluation_results/model_comparison.csv', index=False)
    
    return zero_shot_metrics, fine_tuned_metrics

if __name__ == "__main__":
    # Load and compare models
    fine_tuned_path = "models/finetuned/best"
    zero_shot_metrics, fine_tuned_metrics = compare_models(fine_tuned_path) 