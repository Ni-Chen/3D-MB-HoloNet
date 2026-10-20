import math
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt


def _matlab_complex(array):
    if array.dtype.fields:
        fields = array.dtype.fields
        real_name = next((name for name in fields if name.lower() == "real"), None)
        imag_name = next((name for name in fields if name.lower() == "imag"), None)
        if real_name is not None and imag_name is not None:
            return array[real_name] + 1j * array[imag_name]
    return array


def load_holo(data_path, start_index=None, data_num=None):
    with h5py.File(data_path, "r") as mat:
        data = np.asarray(mat["data"][()]).transpose().astype(np.float32)
        label = np.asarray(mat["label"][()]).transpose().astype(np.float32)
        label = label.transpose(0, 3, 1, 2)
        otf = _matlab_complex(np.asarray(mat["otf3d"][()])).transpose()
        otf = otf.transpose(2, 0, 1)
        if not np.iscomplexobj(otf):
            raise ValueError("otf3d must contain complex-valued transfer data")
        otf = otf[np.newaxis, ...].astype(np.complex64)

    if (start_index is None) != (data_num is None):
        raise ValueError("start_index and data_num must be provided together")
    if start_index is not None:
        start = max(0, start_index - 1)
        data = data[start:start + data_num]
        label = label[start:start + data_num]

    return np.ascontiguousarray(data), np.ascontiguousarray(label), otf


def norm_data(data):
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std == 0:
        raise ValueError("cannot normalize holograms with zero standard deviation")
    return (data - mean) / std, mean, std


def norm_data_mu(data, mean, std):
    if std == 0:
        raise ValueError("normalization standard deviation must be nonzero")
    return (data - mean) / std


def dataset_name(args):
    return (
        f"{args.obj_type}_Nz{args.Nz}_ppv{args.ppv}_dz{args.dz}"
        f"_L{args.layer_num}_B{args.batch_size}_N{args.data_num}"
        f"_lr{args.lr_max}_G{args.gamma}_R{args.reg}"
    )


def plotcube(volume, title, file_name):
    low = np.min(volume)
    high = np.max(volume)
    if high > low:
        volume = (volume - low) / (high - low)
    depth = volume.shape[0]
    columns = depth if depth <= 10 else math.ceil(math.sqrt(depth))
    rows = math.ceil(depth / columns)
    fig = plt.figure(figsize=(columns * 5, rows * 5 + 0.5))
    fig.suptitle(title, y=1)
    for index in range(depth):
        axis = fig.add_subplot(rows, columns, index + 1)
        axis.set_title(f"z {index + 1}")
        axis.imshow(volume[index], aspect="equal")
    fig.tight_layout()
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name)
    plt.close(fig)
