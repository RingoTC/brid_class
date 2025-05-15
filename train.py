import os
import yaml
import datetime
import argparse
import numpy as np
import torch
import shutil
from sklearn.metrics import roc_auc_score
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from utils import get_device
from dataset import load_dataset, ImageClassificationDataset


def train_model(dataset_info, model_name="google/vit-base-patch16-224",
                num_epochs=100, batch_size=32, learning_rate=2e-5, output_dir=None,
                kfold_config=None, fold_idx=None, patience=3):
    """Train the ViT model
    
    Args:
        split_data: Dictionary with split data
        yaml_content: Dataset metadata
        model_name: Name of the pretrained model to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        output_dir: Output directory for saving models
        kfold_config: K-fold configuration
        fold_idx: Current fold index (0-based)
        
    Returns:
        model: Trained model
        processor: Image processor
    """
    device = get_device()
    
    # Get dataset info
    num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names']
    
    # Generate output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir or "models", timestamp)
    if kfold_config and kfold_config.get('enabled', False) and fold_idx is not None:
        print(f"\nTraining fold {fold_idx+1}/{kfold_config.get('k', 5)}")
        output_dir = os.path.join(output_dir, f"fold_{fold_idx+1}")
    print(f"\nNumber of classes: {num_classes}")
    
    # Initialize model and processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Initialize datasets with processor
    train_dataset = dataset_info['train_dataset']
    val_dataset = dataset_info['val_dataset']
    train_dataset.transform = processor
    val_dataset.transform = processor
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        push_to_hub=False,
        remove_unused_columns=False,
        logging_steps=50,
    )
    
    # Add early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.0
    )
    
    def compute_metrics(eval_pred):
        # Handle tuple output from ViT model
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids

        # Debug information
        print("\nDebug Info:")
        print(f"Logits shape: {logits.shape}")
        print(f"Number of actual classes: {num_classes}")
        
        # Ensure logits match the actual number of classes
        if logits.shape[1] != num_classes:
            print(f"Warning: Logits dimension ({logits.shape[1]}) does not match number of classes ({num_classes})")
            # Only use the first num_classes logits
            logits = logits[:, :num_classes]
            print(f"Truncated logits shape: {logits.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        print(f"Unique labels: {np.unique(labels)}")
        
        # Apply softmax with numerical stability
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Calculate predictions and accuracy
        predictions = np.argmax(probas, axis=1)
        predictions = predictions.astype(labels.dtype)  # Ensure same dtype as labels
        accuracy = np.mean(predictions == labels)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions dtype: {predictions.dtype}")
        print(f"Unique predictions: {np.unique(predictions)}")
        print(f"Match count: {np.sum(predictions == labels)} out of {len(labels)}")
        
        # Convert labels to one-hot format
        n_classes = logits.shape[1]
        labels_one_hot = np.zeros((len(labels), n_classes))
        labels_one_hot[np.arange(len(labels)), labels] = 1
        
        # Calculate AUC for each class
        auc_scores = []
        for i in range(n_classes):
            try:
                # For classes with no samples, assign AUC of 0.5 (random chance)
                if len(np.unique(labels_one_hot[:, i])) <= 1:
                    auc_scores.append(0.5)
                else:
                    auc = roc_auc_score(labels_one_hot[:, i], probas[:, i])
                    auc_scores.append(auc)
            except ValueError as e:
                print(f"Warning: Could not calculate AUC for class {i}: {e}")
                # If calculation fails, treat as random chance
                auc_scores.append(0.5)
        
        # Calculate macro average AUC across all classes
        macro_auc = np.mean(auc_scores)
        
        return {
            "accuracy": accuracy,
            "auc": macro_auc
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    # Train the model
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train()
    
    # Save the best model
    best_model_path = os.path.join(output_dir, "best")
    trainer.save_model(best_model_path)
    processor.save_pretrained(best_model_path)
    
    # Return metrics from trainer
    metrics = trainer.evaluate()
    print(f"Validation metrics: {metrics}")
    
    return model, processor, metrics

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default=None,
                      help='Override dataset name from config')
    parser.add_argument('--model', type=str, default=None,
                      help='Override model from config')
    parser.add_argument('--sample-ratio', type=float, default=None,
                      help='Override sample ratio from config')
    parser.add_argument('--kfold', action='store_true',
                      help='Enable k-fold cross validation (overrides config)')
    parser.add_argument('--no-kfold', action='store_true',
                      help='Disable k-fold cross validation (overrides config)')
    parser.add_argument('--patience', type=int, default=3,
                      help='Number of evaluations with no improvement after which training will be stopped')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    dataset_name = args.dataset or config['dataset']['name']
    model_name = args.model or config['model']['name']
    sample_ratio = args.sample_ratio if args.sample_ratio is not None else float(config['dataset']['sample_ratio'])
    output_dir = config['model']['output_dir']
    train_ratio = float(config['dataset']['train_ratio'])
    val_ratio = float(config['dataset']['val_ratio'])
    base_dir = config['dataset']['base_dir']
    num_epochs = int(config['training']['num_epochs'])
    batch_size = int(config['training']['batch_size'])
    learning_rate = float(config['training']['learning_rate'])
    seed = int(config['training']['seed']) 
    patience = args.patience
    
    # K-fold settings
    kfold_config = config['kfold'].copy()  # Make a copy to avoid modifying original
    # Convert numeric parameters
    if 'k' in kfold_config:
        kfold_config['k'] = int(kfold_config['k'])
    # Command line arguments override config
    if args.kfold:
        kfold_config['enabled'] = True
    elif args.no_kfold:
        kfold_config['enabled'] = False
        
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare dataset
    print(f"Preparing {dataset_name} dataset with sample ratio {sample_ratio}...")
    print(f"{'K-fold cross validation enabled' if kfold_config['enabled'] else 'Standard training'}")
    
    # Load or prepare dataset
    # Load dataset
    dataset_info = load_dataset(
        dataset_name,
        base_dir=base_dir,
        transform=None,  # transformations are handled in training class
        kfold_config=kfold_config
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train model based on k-fold setting
    if kfold_config['enabled']:
        print(f"\nStarting {kfold_config['k']}-fold cross validation training...")
        
        # For storing metrics across folds
        all_metrics = []
        best_fold = 0
        best_accuracy = 0
        best_auc = 0
        
        for fold_idx in range(kfold_config['k']):
            # Get dataset for current fold
            fold_dataset_info = load_dataset(
                dataset_name,
                base_dir=base_dir,
                transform=None,
                kfold_config={**kfold_config, 'fold': fold_idx}
            )
            
            # Train model for this fold
            model, processor, metrics = train_model(
                fold_dataset_info,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=output_dir,
                kfold_config=kfold_config,
                fold_idx=fold_idx,
                patience=patience
            )
            
            all_metrics.append(metrics)
            
            # Track best fold (using AUC as primary metric)
            if metrics['eval_auc'] > best_auc:
                best_accuracy = metrics['eval_accuracy']
                best_auc = metrics['eval_auc']
                best_fold = fold_idx
        
        # Compute average metrics across folds
        avg_accuracy = np.mean([m['eval_accuracy'] for m in all_metrics])
        avg_auc = np.mean([m['eval_auc'] for m in all_metrics])
        
        print(f"\nK-fold cross validation completed!")
        print(f"Average metrics across {kfold_config['k']} folds:")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  AUC: {avg_auc:.4f}")
        print(f"Best fold: {best_fold+1}")
        print(f"  Accuracy: {best_accuracy:.4f}")
        print(f"  AUC: {best_auc:.4f}")
        print(f"Models saved in {output_dir or f'models/{dataset_name}'}/kfold_{timestamp}/")
        
        # If not saving all folds, remove the non-best folds
        if not kfold_config['save_all_folds']:
            best_fold_dir = f"{output_dir or f'models/{dataset_name}'}/kfold_{timestamp}/fold_{best_fold+1}"
            for i in range(kfold_config['k']):
                if i != best_fold:
                    fold_dir = f"{output_dir or f'models/{dataset_name}'}/kfold_{timestamp}/fold_{i+1}"
                    if os.path.exists(fold_dir):
                        shutil.rmtree(fold_dir)
    else:
        # Standard training
        model, processor, metrics = train_model(
            dataset_info,
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            patience=patience
        )
        
        print(f"\nTraining completed! Model saved in {output_dir or f'models/{dataset_name}'}/{timestamp}/best/")
        print(f"Validation metrics:")
        print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"  AUC: {metrics['eval_auc']:.4f}")

if __name__ == "__main__":
    main()
