# test.py
import os
import yaml
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

# --- project imports ---------------------------------------------------------
from src.dataset import create_dataset, collate_fn
from src.models import BiLSTMRegressor

# ----------------------------------------------------------------------------- 
# Utility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 53):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def run_test(model, dataloader, loss_fn, device):
    """
    Returns the average (mean) loss over *valid* batches.
    If you used MSE during training we additionally return RMSE.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    is_mse = isinstance(loss_fn, nn.MSELoss)

    for batch in dataloader:
        if batch is None:          # collate_fn found no valid segments
            continue

        ppg  = [seg.to(device) for seg in batch["ppg"]]
        y    = batch["label"].to(device)
        slen = batch["segment_lengths"]

        y_hat = model(ppg, slen)
        loss  = loss_fn(y_hat, y)

        bs = y.size(0)
        running_loss += loss.detach().item() * bs
        num_samples += bs

        print("Real SDNN     :", [f"{v:.3f}" for v in y.cpu().tolist()])
        print("Predicted SDNN:", [f"{v:.3f}" for v in y_hat.cpu().tolist()])
        print ("")

    if num_samples == 0:
        raise RuntimeError("No usable samples in the test set.")

    mean_loss = running_loss / num_samples

    return mean_loss ** 0.5 if is_mse else mean_loss

# ----------------------------------------------------------------------------- 
# Main
# -----------------------------------------------------------------------------
def main():
    CFG_PATH   = "config/config.yaml"
    CKPT_PATH  = "checkpoints/best.pt"

    # ----------------- load config & checkpoint ------------------------------
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- dataset & dataloader ----------------------------------
    data_root = (
        "Data/Dataset_processed" if cfg["generate_dataset"]["quality_assessment"]
        else "Data/Dataset_raw"
    )
    *_ , test_data = create_dataset(data_root, random_seed=cfg["seed"])

    test_loader = DataLoader(
        test_data,
        batch_size = 4,
        shuffle    = False,
        collate_fn = collate_fn,
        pin_memory = True
    )

    # ----------------- model --------------------------------------------------
    model = BiLSTMRegressor(
        cnn_out_dim      = cfg["model"]["cnn_out_dim"],
        lstm_hidden_dim  = cfg["model"]["lstm_hidden_dim"],
        lstm_layers      = cfg["model"]["lstm_layers"],
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint '{CKPT_PATH}' not found")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')} "
          f"(val_loss={ckpt.get('val_loss','?'):.4f})")

    # ----------------- loss ---------------------------------------------------
    loss_map = {
        "mse"  : nn.MSELoss(),
        "mae"  : nn.L1Loss(),
        "huber": nn.SmoothL1Loss(),   # Huber with delta=1
    }
    loss_fn = loss_map[cfg["training"]["loss"].lower()]

    # ----------------- evaluation --------------------------------------------
    test_loss = run_test(model, test_loader, loss_fn, device)


    print(f"Test {cfg['training']['loss'].upper()} : {test_loss:.4f}")


if __name__ == "__main__":
    main()