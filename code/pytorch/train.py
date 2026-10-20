import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from holonet import MOHoloNet
from utils import dataset_name, load_holo, norm_data, norm_data_mu


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PyTorch MB-HoloNet")
    parser.add_argument("--obj_type", default="sim")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_num", type=int, default=1000)
    parser.add_argument("--ppv", default="5e-03")
    parser.add_argument("--Nxy", type=int, default=32)
    parser.add_argument("--Nz", type=int, default=3)
    parser.add_argument("--dz", default="1200um")
    parser.add_argument("--data_path")
    parser.add_argument("--model_dir")
    return parser.parse_args()


def run_epoch(model, loader, otf, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    for hologram, label in loader:
        hologram = hologram.to(device)
        label = label.to(device)
        batch_otf = otf.expand(hologram.shape[0], -1, -1, -1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction, symmetry_loss = model(hologram, batch_otf, return_aux=True)
        loss = (prediction - label).square().mean()
        loss = loss + symmetry_loss + model.regularization_loss()
        if training:
            loss.backward()
            optimizer.step()
            model.project_nonnegative_parameters()
        total_loss += loss.detach().item() * hologram.shape[0]
        sample_count += hologram.shape[0]
    return total_loss / sample_count


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parents[1]
    data_path = Path(args.data_path) if args.data_path else (
        repo_dir / "code" / "data" /
        f"{args.obj_type}_Nz{args.Nz}_ppv{args.ppv}_50db_dz{args.dz}.mat"
    )
    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    name = dataset_name(args)
    model_path = model_dir / f"holonet_{name}.pt"
    stats_path = model_dir / f"{name}.mat"

    data, labels, otf_array = load_holo(data_path, 1, args.data_num)
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    train_data, data_mean, data_std = norm_data(train_data)
    val_data = norm_data_mu(val_data, data_mean, data_std)
    scipy.io.savemat(
        stats_path,
        {
            "data_mean": data_mean,
            "data_std": data_std,
            "data_max": np.max(train_data),
            "data_min": np.min(train_data),
        },
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_labels)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOHoloNet(
        args.Nxy, args.Nxy, args.Nz, args.layer_num, args.gamma, args.reg
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    start_epoch = args.start_epoch
    best_loss = float("inf")
    if start_epoch:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint.get("best_loss", best_loss)

    otf = torch.from_numpy(otf_array).to(device)
    for epoch in range(start_epoch, args.epochs):
        progress = epoch / max(args.epochs, 1)
        lr = args.lr_min + (args.lr_max - args.lr_min) * (1 - progress) ** 2
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_loss = run_epoch(model, train_loader, otf, device, optimizer)
        val_loss = run_epoch(model, val_loader, otf, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"lr={lr:.6g} train_loss={train_loss:.6g} val_loss={val_loss:.6g}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                },
                model_path,
            )


if __name__ == "__main__":
    main()
