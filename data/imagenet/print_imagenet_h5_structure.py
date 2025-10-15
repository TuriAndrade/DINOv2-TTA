import h5py
from pathlib import Path

H5_PATH = Path("./imagenet.h5")  # path to your HDF5 file

def print_h5_structure(file_path: Path):
    with h5py.File(file_path, "r") as f:
        def print_attrs(name, obj, indent=0):
            ind = "  " * indent
            if isinstance(obj, h5py.Dataset):
                print(f"{ind}- Dataset: {name} shape={obj.shape} dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{ind}+ Group: {name}")
            if obj.attrs:
                for k, v in obj.attrs.items():
                    print(f"{ind}    @attr {k} = {v}")

        def visit(name, obj):
            depth = name.count("/")
            print_attrs(name, obj, depth)

        print(f"File: {file_path}")
        print(f"Attributes: {dict(f.attrs)}")
        f.visititems(visit)

print_h5_structure(H5_PATH)
