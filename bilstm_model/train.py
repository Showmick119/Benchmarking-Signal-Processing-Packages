# Training loop code
import os
import yaml
import random
import torch
import tqdm
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import create_dataset, collate_fn
from models import BiLSTMRegressor, FullyConnected

def set_seed(seed: int = 53):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_validation(model, val_loader, loss_fn, device, test = False):
    model.eval()
    val_loss = 0.0
    num_samples = 0
    is_mse = isinstance(loss_fn, nn.MSELoss)

    for batch in val_loader:
        if batch is None:
            continue

        ppg = [seg.to(device) for seg in batch['ppg']]
        label = batch["label"].to(device)
        seg_l = batch["segment_lengths"]

        out = model(ppg, seg_l)
        loss = loss_fn(out, label)

        if out.isnan().any() or out.isinf().any():
            print ("NAN or INF detected in output during validation")
        
        #bs = label.size(0)
        val_loss += loss.detach().item() #* bs
        num_samples += 1

        # print ('\nPredicted_Val:', out.detach().cpu().numpy())
        # print("Label_val:", label.detach().cpu().numpy())
        # print ('Loss_val:', loss.item())
        # print (f"Output mean_val: {out.mean().item()}, std: {out.std().item()}") # Low std means nearly-constant output
        # print ('')
        
        if test:
            print("Real SDNN     :", [f"{v:.3f}" for v in label.cpu().tolist()])
            print("Predicted SDNN:", [f"{v:.3f}" for v in out.cpu().tolist()])

    if num_samples == 0:
        raise RuntimeError("No usable samples in the validation set.")
    
    mean_loss = val_loss / num_samples

    return mean_loss ** 0.5 if is_mse else mean_loss


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(
    project="HRV",
    config=config,
    )

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data ----------------------------------------------------------------------#
    # if config['generate_dataset']['quality_assessment']:
    #     data_dir = 'Data/Dataset_processed'
    # else:
    #     data_dir = 'Data/Dataset_raw'

    data_dir = config['training_data']['path']
    
    train_data, val_data, test_data = create_dataset(data_dir, random_seed = config["seed"])

    train_loader = DataLoader(
        train_data,
        batch_size = config["training"]["batch_size"],
        num_workers = 1,
        shuffle = False,     # Shuffles samples but keeps segments in order
        collate_fn = collate_fn,
        pin_memory = True,
        drop_last = True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size = 4, 
        shuffle = True, 
        collate_fn = collate_fn, 
        pin_memory = True,
        drop_last = True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size = 4, 
        shuffle = True, 
        collate_fn = collate_fn, 
        pin_memory = True,
        drop_last = True
    )

    # Model ---------------------------------------------------------------------#
    model = BiLSTMRegressor(
        cnn_out_dim = config['model']['cnn_out_dim'],
        lstm_hidden_dim = config['model']['lstm_hidden_dim'],
        lstm_layers = config['model']['lstm_layers'],
        dropout = config['model']['dropout'],
        pooling = config['model']['pooling'],
        attention_hidden_dim = config['model']['attention_hidden_dim']
    ).to(device)

    print ('\n----- Current Model Parameters -----\n')
    print (f"CNN out dim: {config['model']['cnn_out_dim']}")
    print (f"LSTM hidden dim: {config['model']['lstm_hidden_dim']}")
    print (f"Dropout: {config['model']['dropout']}")
    #print (f"Encoder Pooling: {config['model']['pooling']}")
    #if config['model']['pooling'] == 'attention':
    #    print (f"Attention hidden dim: {config['model']['attention_hidden_dim']}")
    print ('\n-----\n')

    # model = FullyConnected(
    #     cnn_out_dim = config['model']['cnn_out_dim'],
    #     hidden_dims = config['model']['hidden_dims'],
    #     layers = config['model']['layers'],
    #     dropout = config['model']['dropout']
    # ).to(device)

    # Loss ----------------------------------------------------------------------#
    loss_map = {"mse": nn.MSELoss(),
                "mae": nn.L1Loss(),
                "huber": nn.SmoothL1Loss()      # HuberLoss with delta = 1
                }
    loss_fn = loss_map[config["training"]["loss"].lower()]
    is_mse = isinstance(loss_fn, nn.MSELoss)

    # Optimizer -----------------------------------------------------------------#
    optim_map = {"adam": torch.optim.Adam,
                 "adamw": torch.optim.AdamW,
                 "sgd": torch.optim.SGD}
    optim_cls = optim_map[config["training"]["optimizer"].lower()]
    optimizer = optim_cls(model.parameters(),
                          lr = config["training"]["learning_rate"],
                          weight_decay = config["training"].get("weight_decay", 0.0))
    
    # Scheduler -----------------------------------------------------------------#
    scheduler = None
    sched_cfg = config["training"].get("scheduler")
    if sched_cfg:
        if sched_cfg["type"].lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size = sched_cfg["step_size"],
                gamma = sched_cfg.get("gamma", 0.1)
            )
        elif sched_cfg["type"].lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode = "min",
                factor = sched_cfg.get("factor", 0.5),
                patience = sched_cfg.get("patience", 5)
            )

    # Checkpoint ----------------------------------------------------------------#
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float('inf')
    start_epoch = 0

    # Training Loop -------------------------------------------------------------#
    epochs = config["training"]["epochs"]
    
    # DEBUG ---------------------------------------------------------------------#
    # NOTE: Code line 187-225 overfits the model to a small batch of same examples.
    # single_batch = next(iter(train_loader))
    # ppg_segments = [seg.to(device) for seg in single_batch['ppg']]
    # label = single_batch['label'].to(device)
    # segment_lengths = single_batch['segment_lengths']

    # print(f"\n[INFO] Overfitting on single batch: label = {label.tolist()}\n")

    # for epoch in tqdm.trange(epochs, desc="Overfitting", unit="epoch"):
    #     model.train()
    #     optimizer.zero_grad()

    #     output = model(ppg_segments, segment_lengths)
    #     loss = loss_fn(output, label)
    #     loss_value = loss.item()

    #     #print(f"[Epoch {epoch+1:03d}] Loss: {loss_value:.4f} | Output: {output.detach().cpu().numpy()}")

    #     loss.backward()

    #     # Optional: print gradient norms
    #     total_grad_norm = 0.0
    #     for name, p in model.named_parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             total_grad_norm += param_norm.item() ** 2
    #     total_grad_norm = total_grad_norm ** 0.5

    #     #torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    #     optimizer.step()

    #     wandb.log({
    #         "batch/overfit_loss": loss_value**0.5 if is_mse else loss_value,
    #         "batch/grad_norm": total_grad_norm
    #     })
    
    # model.eval()
    # output = model(ppg_segments, segment_lengths)
    # print (f'\nLabel: {label}')
    # print (f'Output: {output}')
    # DEBUG -------------------------------------------------------------------#

    # NOTE: Enable code line 229-345 to train the model on entire data.
    try:
        for epoch in tqdm.trange(start_epoch, epochs, desc="Training", unit="epoch"):
            with tqdm.tqdm(train_loader, desc=f'epoch {epoch+1}', unit='batch', total=len(train_loader), position=0, leave=True) as batch_iterator:
                model.train()
                running_loss = 0.0
                num_samples = 0

                for i, batch in enumerate(batch_iterator, start = 1):
                    if batch is None:
                        continue

                    optimizer.zero_grad()

                    ppg_segments = [seg.to(device) for seg in batch['ppg']]  # Shape: (batch_size, seq_length, data_points) -> data_points are padded to max length
                    #ecg_segments = batch["ecg"] # (optional) because we don't need ecg_signal for training 
                    label = batch["label"].to(device)
                    segment_lengths = batch["segment_lengths"]  # list of lists (batch_size, segs, len)

                    output = model(ppg_segments, segment_lengths)
                    loss = loss_fn(output, label)
                    loss_value = loss.detach().item()                 

                    loss.backward()
                    total_grad_norm = 0.0
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    
                    total_grad_norm = total_grad_norm ** 0.5 # L2 norm

                    torch.nn.utils.clip_grad_norm(model.parameters(), 3.0)      # Gradient clipping with max_norm == 1.0 (0.5 - 5.0)
                    optimizer.step()
                    
                    # Loss calculation
                    bs = label.size(0)                      
                    running_loss += loss_value #* bs           # Sum of squares of losses in a batch
                    num_samples += 1 #bs

                    # Log batch-level metrics
                    wandb.log({
                        "batch/train_loss": loss_value**0.5 if is_mse else loss_value,
                        "batch/learning_rate": optimizer.param_groups[0]['lr'],
                        "batch/grad_norm": total_grad_norm
                    }, commit = False)

                    batch_iterator.set_postfix(
                        {
                            "Mean Loss": (running_loss / num_samples)**0.5 if is_mse else (running_loss / num_samples),
                            "Current Loss": (loss.detach().item())**0.5 if is_mse else (loss.detach().item())
                        }
                    )

                if (epoch != 0) and (epoch % 25 == 0):
                        print ('\nOutput:', output.detach().cpu().numpy())
                        print("Label:", label.detach().cpu().numpy())
                        print ('Loss:', loss.item())
                        print (f"Output mean_val: {output.mean().item()}, std: {output.std().item()}")  # Low std means nearly-constant output
                        print ('')

                # Calculate epoch-level training metrics
                epoch_train_loss = (running_loss / num_samples)**0.5 if is_mse else (running_loss / num_samples)

                # ------ Validation ------
                val_loss = run_validation(model, val_loader, loss_fn, device)
                print (f'[Epoch {epoch+1} val_loss: {val_loss:.4f}]\n')

                # ------- Test Log -------
                if (epoch != 0) and (epoch % 50 == 0):
                    test_loss = run_validation(model, test_loader, loss_fn, device, test = True)
                    print (f'[Epoch {epoch+1} test_loss: {test_loss:.4f}]\n')
                
                # Log epoch-level metrics
                wandb.log({
                    "epoch/train_loss": epoch_train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch+1
                })

                # ------ Scheduler Step ------
                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # ------ Checkpoint ------
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "epoch": epoch+1,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "val_loss": best_val
                    }, os.path.join(ckpt_dir, "best.pt"))
                    print (f"Saved best model (val_loss={best_val:.4f})")

        # Save the latest model
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss": val_loss
        }, os.path.join(ckpt_dir, "latest.pt"))
        print (f"Saved latest model (val_loss={val_loss:.4f})")

    except (KeyboardInterrupt, StopIteration):
        print (f"[INFO] Interrupted --- Saving last checkpoint ...")
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict()},
                    os.path.join(ckpt_dir, "latest.pt"))
    
    # Test the model
    test_loss = run_validation(model, test_loader, loss_fn, device, test=True)
    print (f'\nTest_loss: {test_loss:.4f}]\n')

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()