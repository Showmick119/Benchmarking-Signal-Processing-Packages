# train.py - Training script for PPG Segment Quality Classifier
import os
import yaml
import random
import torch
import tqdm
import wandb
import numpy as np
import argparse
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

# project imports
from .dataset import create_k_fold_datasets, create_data_loaders, collate_fn
from .classifier import SegmentCNNClassifier, EarlyStopping


def set_seed(seed: int = 2053) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(model, dataloader, loss_fn, device):
    """
    Evaluate model and return comprehensive metrics.
    
    Returns:
        dict: Dictionary containing loss, accuracy, f1, precision, recall, auc
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    num_batches = 0
    
    for batch in dataloader:
        if batch is None:
            continue
        
        signals = batch['signal'].to(device)
        labels = batch['label'].to(device).float()
        lengths = batch['lengths'].to(device)
        
        probabilities = model(signals, lengths)
        loss = loss_fn(probabilities, labels)
        
        predictions = (probabilities >= 0.5).float()
        
        running_loss += loss.item()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        num_batches += 1
    
    if num_batches == 0:
        raise RuntimeError("No valid batches found in evaluation.")
    
    avg_loss = running_loss / num_batches
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = 0.0  # Handle case where only one class is present
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


def train_single_fold(model, train_loader, val_loader, config, device, fold_idx, run_name):
    """
    Train model for a single fold.
    """
    loss_fn = nn.BCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        min_lr=config['training']['scheduler']['min_lr']
    )

    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )

    best_val_f1 = 0.0
    best_metrics = {}
    epochs = config['training']['epochs']

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        with tqdm.tqdm(train_loader, desc=f'Fold {fold_idx+1} Epoch {epoch+1}', 
                      unit='batch', leave=False) as batch_iterator:
            
            for batch_idx, batch in enumerate(batch_iterator):
                if batch is None:
                    continue
                
                optimizer.zero_grad()
                
                signals = batch['signal'].to(device)
                labels = batch['label'].to(device).float()
                lengths = batch['lengths'].to(device)
                
                probabilities = model(signals, lengths)
                loss = loss_fn(probabilities, labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                batch_iterator.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{running_loss/num_batches:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        if num_batches == 0:
            print(f"No valid batches in epoch {epoch+1}")
            continue

        epoch_train_loss = running_loss / num_batches

        val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        scheduler.step(val_metrics['loss'])

        wandb.log({
            f'fold_{fold_idx+1}/epoch': epoch + 1,
            f'fold_{fold_idx+1}/train_loss': epoch_train_loss,
            f'fold_{fold_idx+1}/val_loss': val_metrics['loss'],
            f'fold_{fold_idx+1}/val_f1': val_metrics['f1'],
            f'fold_{fold_idx+1}/val_accuracy': val_metrics['accuracy'],
            f'fold_{fold_idx+1}/learning_rate': optimizer.param_groups[0]['lr']
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch + 1

        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        print(f"Epoch {epoch+1}: Val F1={val_metrics['f1']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}, Val Loss={val_metrics['loss']:.4f}")
    
    return best_metrics


def main():
    """Main training function with K-fold cross validation."""

    parser = argparse.ArgumentParser(description='Train PPG Segment Quality Classifier')
    parser.add_argument('--config', type=str, default="config/segment_config.yaml",
                       help='Path to configuration file (default: config/segment_config.yaml)')
    args = parser.parse_args()

    config_path = args.config
    print(f"Using config: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = "segment_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    wandb_config = config.get('wandb', {})

    run_name = f"segment_classifier_k{config['data']['k_folds']}_fold_cv"
    
    wandb.init(
        project=wandb_config.get('project', 'HRV_Segment_Classification'),
        entity=wandb_config.get('entity', None),
        name=run_name,
        config={
            'learning_rate': config['training']['learning_rate'],
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'dropout': config['model']['dropout'],
            'k_folds': config['data']['k_folds'],
            'architecture': 'CNN'
        },
        mode=wandb_config.get('mode', 'online'),
        notes=wandb_config.get('notes', 'PPG segment quality classifier'),
        tags=['ppg', 'classification', 'cnn']
    )
    
    print(f"Weights & Biases initialized in {wandb_config.get('mode', 'online')} mode")
    print(f"View your run at: {wandb.run.url}")

    print("Preparing datasets...")
    test_dataset, fold_datasets = create_k_fold_datasets(
        data_path=config['data']['path'],
        k_folds=config['data']['k_folds'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['seed']
    )

    fold_results = []
    
    for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{config['data']['k_folds']}")
        print(f"{'='*60}")

        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset,
            batch_size=config['training']['batch_size']
        )

        model = SegmentCNNClassifier(
            input_channels=config['model']['input_channels'],
            hidden_channels=config['model']['hidden_channels'],
            kernel_sizes=config['model']['kernel_sizes'],
            dropout=config['model']['dropout'],
            pool_size=config['model']['pool_size']
        ).to(device)

        fold_metrics = train_single_fold(
            model, train_loader, val_loader, config, device, fold_idx, run_name
        )
        
        fold_results.append(fold_metrics)

        fold_path = os.path.join(output_dir, f"fold_{fold_idx+1}_best.pt")
        torch.save({
            'fold': fold_idx + 1,
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': fold_metrics
        }, fold_path)
        
        print(f"Fold {fold_idx+1} Results:")
        print(f"  Best F1: {fold_metrics['f1']:.4f}")
        print(f"  Best Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  Best AUC: {fold_metrics['auc']:.4f}")

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    cv_stats = {}
    for metric in ['f1', 'accuracy', 'precision', 'recall', 'auc', 'loss']:
        values = [result[metric] for result in fold_results]
        cv_stats[f'cv_{metric}_mean'] = np.mean(values)
        cv_stats[f'cv_{metric}_std'] = np.std(values)
        print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    best_fold_idx = np.argmax([result['f1'] for result in fold_results])
    best_fold_metrics = fold_results[best_fold_idx]
    
    print(f"\nBest Fold: {best_fold_idx + 1}")
    print(f"Best F1 Score: {best_fold_metrics['f1']:.4f}")

    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    best_model_path = os.path.join(output_dir, f"fold_{best_fold_idx+1}_best.pt")
    checkpoint = torch.load(best_model_path, map_location=device)
    
    final_model = SegmentCNNClassifier(
        input_channels=config['model']['input_channels'],
        hidden_channels=config['model']['hidden_channels'],
        kernel_sizes=config['model']['kernel_sizes'],
        dropout=config['model']['dropout'],
        pool_size=config['model']['pool_size']
    ).to(device)
    
    final_model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0
    )
    
    test_metrics = evaluate_model(final_model, test_loader, nn.BCELoss(), device)
    
    print("Test Set Results:")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")

    final_model_path = os.path.join(output_dir, "best_segment_classifier.pt")
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': config,
        'cv_results': fold_results,
        'cv_stats': cv_stats,
        'test_metrics': test_metrics,
        'best_fold': best_fold_idx + 1
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
    
    wandb.log({
        'cv_f1_mean': cv_stats['cv_f1_mean'],
        'cv_f1_std': cv_stats['cv_f1_std'],
        'cv_accuracy_mean': cv_stats['cv_accuracy_mean'],
        'cv_accuracy_std': cv_stats['cv_accuracy_std'],
        'cv_loss_mean': cv_stats['cv_loss_mean'],

        'test_f1': test_metrics['f1'],
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
        'test_loss': test_metrics['loss'],

        'best_fold': best_fold_idx + 1,
        'total_parameters': sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    })

    fold_data = []
    for i, result in enumerate(fold_results):
        fold_data.append([i + 1, result['f1'], result['accuracy']])
    
    fold_table = wandb.Table(data=fold_data, columns=["Fold", "F1_Score", "Accuracy"])
    wandb.log({
        "cv_performance_by_fold": wandb.plot.line(fold_table, "Fold", "F1_Score", title="F1 Score by Fold"),
        "cv_accuracy_by_fold": wandb.plot.line(fold_table, "Fold", "Accuracy", title="Accuracy by Fold")
    })
    
    print(f"\nFinal Results Summary:")
    print(f"Cross-Validation F1: {cv_stats['cv_f1_mean']:.4f} ± {cv_stats['cv_f1_std']:.4f}")
    print(f"Test Set F1: {test_metrics['f1']:.4f}")
    print(f"Test Set Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Best Fold: {best_fold_idx + 1}")
    print(f"View results at: {wandb.run.url}")
    
    wandb.finish()
    
    print(f"\nTraining completed! Best model saved as: {final_model_path}")


if __name__ == "__main__":
    main()