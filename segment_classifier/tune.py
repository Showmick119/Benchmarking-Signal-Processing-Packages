# tune.py - Hyperparameter tuning for PPG Segment Quality Classifier using Optuna
import os
import yaml
import torch
import torch.nn as nn
import wandb
import optuna
import numpy as np
from optuna.integration.wandb import WeightsAndBiasesCallback

# project imports
from .dataset import create_k_fold_datasets, create_data_loaders
from .classifier import SegmentCNNClassifier, EarlyStopping
from .train import train_single_fold, evaluate_model, set_seed


def suggest_hyperparameters(trial):
    """
    Suggest hyperparameters for Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Dictionary of suggested hyperparameters
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, log=False),
        'batch_size': trial.suggest_categorical('batch_size', (16, 32, 64)),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'hidden_channels': trial.suggest_categorical('hidden_channels', (
            (16, 32, 64),
            (32, 64, 128), 
            (64, 128, 256)
        ))
    }


def objective(trial):
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: CV F1 score to maximize
    """
    suggested_params = suggest_hyperparameters(trial)
    
    config_path = "config/segment_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # update config with suggested hyperparameters
    config['training']['learning_rate'] = suggested_params['learning_rate']
    config['training']['batch_size'] = suggested_params['batch_size']
    config['training']['weight_decay'] = suggested_params['weight_decay']
    config['model']['dropout'] = suggested_params['dropout']
    config['model']['hidden_channels'] = suggested_params['hidden_channels']

    set_seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n  Trial {trial.number + 1}:")
    print(f"  Learning Rate: {suggested_params['learning_rate']:.2e}")
    print(f"  Dropout: {suggested_params['dropout']:.3f}")
    print(f"  Batch Size: {suggested_params['batch_size']}")
    print(f"  Weight Decay: {suggested_params['weight_decay']:.2e}")
    print(f"  Hidden Channels: {suggested_params['hidden_channels']}")
    
    # prepare datasets (reuse same splits for fair comparison)
    test_dataset, fold_datasets = create_k_fold_datasets(
        data_path=config['data']['path'],
        k_folds=config['data']['k_folds'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['seed']  # same seed = same splits
    )

    fold_f1_scores = []
    
    for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f"  Fold {fold_idx + 1}/{config['data']['k_folds']}", end=' ')
        
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
        
        try:
            fold_metrics = train_single_fold_silent(
                model, train_loader, val_loader, config, device, fold_idx
            )
            
            fold_f1 = fold_metrics['f1']
            fold_f1_scores.append(fold_f1)
            print(f"F1: {fold_f1:.4f}")
            
        except Exception as e:
            print(f"Failed: {str(e)}")
            # end failed trials immediately
            return 0.0
        
        # prune trial if performing poorly
        trial.report(np.mean(fold_f1_scores), fold_idx)
        if trial.should_prune():
            print(f"Trial pruned after fold {fold_idx + 1}")
            raise optuna.exceptions.TrialPruned()

    mean_cv_f1 = np.mean(fold_f1_scores)
    std_cv_f1 = np.std(fold_f1_scores)
    
    print(f"CV F1: {mean_cv_f1:.4f} ± {std_cv_f1:.4f}")
    
    return mean_cv_f1


def train_single_fold_silent(model, train_loader, val_loader, config, device, fold_idx):
    """
    Train model for a single fold without wandb logging (for tuning).
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
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            optimizer.zero_grad()
            
            signals = batch['signal'].to(device)
            labels = batch['label'].to(device).float()  # ensuring float for BCELoss
            lengths = batch['lengths'].to(device)
            
            probabilities = model(signals, lengths)
            loss = loss_fn(probabilities, labels)
            
            # check for NaN loss
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        # skip if no valid batches
        if num_batches == 0:
            continue

        epoch_train_loss = running_loss / num_batches

        val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        scheduler.step(val_metrics['loss'])
        
        # track best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch + 1
        
        # early stopping check
        if early_stopping(val_metrics['loss'], model):
            break
    
    return best_metrics


def main():
    """Main hyperparameter tuning function."""
    
    print("PPG Segment Classifier Hyperparameter Tuning")
    print("=" * 60)
    
    study_name = "ppg_segment_classifier_tuning"
    storage_name = f"sqlite:///optuna_studies/{study_name}.db"

    os.makedirs("optuna_studies", exist_ok=True)

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)  # more aggressive pruning
    )
    
    print(f"Study: {study_name}")
    print(f"Database: {storage_name}")
    print(f"Objective: Maximize CV F1 Score")
    print(f"Trials: 20 (reduced for faster tuning)")
    print("=" * 60)
    
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="cv_f1_score",
        wandb_kwargs={
            "project": "HRV_Segment_Tuning",
            "name": f"optuna_tuning_{study_name}",
            "tags": ["optuna", "hyperparameter_tuning", "ppg", "segment"]
        }
    )
    
    try:
        study.optimize(
            objective, 
            n_trials=20,  # Reduced from 35 for faster tuning
            callbacks=[wandb_callback],
            show_progress_bar=True
        )
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"CV F1 Score: {trial.value:.4f}")
    print(f"Trial Number: {trial.number}")
    
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        if key == 'learning_rate' or key == 'weight_decay':
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    best_config_path = "config/best_segment_config.yaml"
    
    with open("config/segment_config.yaml", "r") as f:
        best_config = yaml.safe_load(f)

    best_config['training']['learning_rate'] = trial.params['learning_rate']
    best_config['training']['batch_size'] = trial.params['batch_size']
    best_config['training']['weight_decay'] = trial.params['weight_decay']
    best_config['model']['dropout'] = trial.params['dropout']
    best_config['model']['hidden_channels'] = trial.params['hidden_channels']

    best_config['tuning'] = {
        'study_name': study_name,
        'best_trial': trial.number,
        'best_cv_f1': trial.value,
        'total_trials': len(study.trials),
        'tuning_complete': True
    }

    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, indent=2)
    
    print(f"\nBest configuration saved to: {best_config_path}")
    print(f"Use this config for training: python -m segment_classifier.train --config {best_config_path}")

    print(f"\nStudy Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    wandb.finish()
    print(f"\nHyperparameter tuning completed!")


if __name__ == "__main__":
    main() 