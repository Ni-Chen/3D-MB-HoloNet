import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch

from holonet import MOHoloNet
from utils import dataset_name, load_holo, norm_data_mu, plotcube


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch MB-HoloNet")
    parser.add_argument("--obj_type", default="sim")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--output_dir")
    parser.add_argument("--start_index", type=int)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parents[1]
    data_path = Path(args.data_path) if args.data_path else (
        repo_dir / "code" / "data" /
        f"{args.obj_type}_Nz{args.Nz}_ppv{args.ppv}_50db_dz{args.dz}.mat"
    )
    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "model"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"
    name = dataset_name(args)
    model_path = model_dir / f"holonet_{name}.pt"
    stats = scipy.io.loadmat(model_dir / f"{name}.mat")
    mean = float(np.asarray(stats["data_mean"]).squeeze())
    std = float(np.asarray(stats["data_std"]).squeeze())
    start_index = args.start_index if args.start_index is not None else args.data_num
    data, labels, otf_array = load_holo(data_path, start_index, 50)
    data = norm_data_mu(data, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOHoloNet(
        args.Nxy, args.Nxy, args.Nz, args.layer_num, args.gamma, args.reg
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    hologram = torch.from_numpy(data).to(device)
    otf = torch.from_numpy(otf_array).to(device).expand(data.shape[0], -1, -1, -1)
    with torch.no_grad():
        predictions = model(hologram, otf).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(output_dir / f"{name}_data.mat", {"data": data})
    scipy.io.savemat(output_dir / f"{name}_predict.mat", {"predict": predictions})
    scipy.io.savemat(output_dir / f"{name}_gt.mat", {"gt": labels})
    mae = np.mean(np.abs(labels - predictions))
    print(f"Test MAE: {mae:.6g}")
    for index in range(min(1, len(data))):
        plotcube(labels[index], f"Ground truth {name}", output_dir / f"{name}_gt.png")
        plotcube(predictions[index], f"Prediction {name}", output_dir / f"{name}_predict.png")


if __name__ == "__main__":
    main()
