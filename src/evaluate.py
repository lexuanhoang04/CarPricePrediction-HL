import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data_splits(data_path: str):
    """
    Replicates train.py exactly:
    - Read preprocessed CSV
    - X = drop Price
    - y = log1p(Price)
    - train/val/test split with random_state=42 and 70/15/15
    """
    data = pd.read_csv(data_path)

    if "Price" not in data.columns:
        raise ValueError("CSV must contain a 'Price' column.")

    X = data.drop("Price", axis=1)

    # log1p target exactly like train.py
    y_raw = data["Price"].values.astype(np.float32)
    y_log = torch.log(torch.tensor(y_raw, dtype=torch.float32) + 1.0).numpy()
    y_log = pd.Series(y_log)

    # split into train/val/test exactly like train.py
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_log, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_tensor(X: pd.DataFrame, device: torch.device) -> torch.Tensor:
    return torch.tensor(X.values, dtype=torch.float32, device=device)


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}mse": float(mean_squared_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP checkpoint on car price data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed CSV (must include Price).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to MLP .pt checkpoint (state_dict).")
    parser.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden dim (must match training).")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="", help="cpu, cuda, or empty to auto.")
    parser.add_argument("--save_preds_csv", type=str, default="", help="Optional path to save test predictions CSV.")
    args = parser.parse_args()

    # device
    if args.device.strip():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data splits (same as train.py)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_splits(args.data_path)

    # import and build model exactly like train.py
    # train.py uses: from models.neural_network import MLP
    from models.neural_network import MLP

    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=1).to(device)

    # load checkpoint
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device)

    # Handle either raw state_dict or wrapped dicts
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # If saved under DDP, strip "module."
        new_state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state = new_state

    # If user saved {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    # inference on test set (log-space predictions)
    X_test_t = to_tensor(X_test, device=device)

    preds_log = []
    with torch.no_grad():
        for i in range(0, X_test_t.shape[0], args.batch_size):
            batch = X_test_t[i : i + args.batch_size]
            out = model(batch)  # shape [B,1]
            preds_log.append(out.detach().cpu().numpy())

    y_pred_log = np.vstack(preds_log).reshape(-1)

    # metrics in log-space (matches train.py's reported MAE/MSE/R2)
    metrics_log = eval_regression(np.asarray(y_test), y_pred_log, prefix="log_")

    # also compute metrics in original price-space (exp(log)-1)
    y_true_price = np.expm1(np.asarray(y_test).reshape(-1))
    y_pred_price = np.expm1(y_pred_log)
    metrics_price = eval_regression(y_true_price, y_pred_price, prefix="price_")

    # print results
    print("=== Evaluation (MLP) ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test size: {len(y_test)}")
    print("\n-- Log-space (train.py target space) --")
    for k, v in metrics_log.items():
        print(f"{k}: {v:.6f}")

    print("\n-- Price-space (after expm1) --")
    for k, v in metrics_price.items():
        print(f"{k}: {v:.6f}")

    # optional save predictions
    if args.save_preds_csv.strip():
        out_path = Path(args.save_preds_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = X_test.copy()
        df_out["y_true_log"] = np.asarray(y_test).reshape(-1)
        df_out["y_pred_log"] = y_pred_log
        df_out["y_true_price"] = y_true_price
        df_out["y_pred_price"] = y_pred_price
        df_out.to_csv(out_path, index=False)
        print(f"\nSaved predictions to: {out_path}")


if __name__ == "__main__":
    main()
