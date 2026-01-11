import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description="Train a car price prediction model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed data CSV file.")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--model_type", type=str, choices=["linear_regression", "random_forest", "mlp"],
                        default="linear_regression", help="Type of model to train.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training (only for MLP).")

    # Early stopping: ONLY min-delta (no patience)
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Stop if val loss does not improve by at least this amount (MLP only).",
    )

    args = parser.parse_args()

    # Load preprocessed data
    data = pd.read_csv(args.data_path)
    X = data.drop("Price", axis=1)

    # use log price as target (log1p)
    y = data["Price"]
    y = torch.log(torch.tensor(y.values, dtype=torch.float32) + 1.0)  # Add 1 to avoid log(0)
    y = pd.Series(y.numpy())

    # split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train model
    if args.model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    elif args.model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    elif args.model_type == "mlp":
        from models.neural_network import MLP
        input_dim = X_train.shape[1]
        hidden_dim = 64
        output_dim = 1
        model = MLP(input_dim, hidden_dim, output_dim)
        # bring model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # bring data to device
    if args.model_type in ["linear_regression", "random_forest"]:
        model.fit(X_train, y_train)

        # Save trained model
        import joblib
        joblib.dump(model, args.model_output_path)
        print(f"Model saved to {args.model_output_path}")

        # Evaluate (log-space)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        print(f"Validation MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    else:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=100,
            shuffle=True,
        )
        
        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        best_state_dict = None

        for epoch in range(args.epochs):
            for X_batch, y_batch in dataloader:
                model.train()
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_losses.append(val_loss.item())

                # Early stopping ONLY by min_delta:
                # If not improved by at least min_delta compared to best, stop immediately.
                # if (best_val_loss - val_loss.item()) > args.early_stopping_min_delta:
                #     best_val_loss = val_loss.item()
                #     best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # else:
                #     print(
                #         f"Early stopping at epoch {epoch+1}: "
                #         f"val loss {val_loss.item():.6f} did not improve best {best_val_loss:.6f} "
                #         f"by >= {args.early_stopping_min_delta}"
                #     )
                #     break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Restore best model before saving/eval
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        torch.save(model.state_dict(), args.model_output_path)
        print(f"Model saved to {args.model_output_path} (best val loss: {best_val_loss:.6f})")

        # plot loss curve
        import matplotlib.pyplot as plt
        plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
        plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss Curve")
        plt.savefig("src/plot/loss_curve.png")

        # Evaluate (log-space)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            y_pred_tensor = model(X_test_tensor)
            y_pred = y_pred_tensor.cpu().numpy()

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Test MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()
