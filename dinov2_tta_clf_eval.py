from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import argparse
import torch
import os
from dataloaders import HDF5Dataset
from evaluation import ClfEvaluator
from config import dataset_configs, tta_configs, optimizer_configs
from datetime import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv2 TTA linear-eval on ImageNet validation "
        "(defaults allow running with zero flags).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # show defaults automatically
    )  # ArgumentDefaultsHelpFormatter is the stdlib helper for printing defaults :contentReference[oaicite:1]{index=1}

    p.add_argument(
        "--dataset-config",
        type=str,
        default="imagenet/val",
        help="Config template for the image dataset, e.g. 'imagenet/train'.",
    )
    p.add_argument(
        "--tta-config",
        type=str,
        default="eata",
        help="Config template for the TTA method.",
    )
    p.add_argument(
        "--optim-config",
        type=str,
        default="norm",
        help="Config for the optimizer.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default="./results",
        help="Parent dir where to save the evaluator config and results.",
    )
    p.add_argument(
        "--out-name",
        type=Path,
        default=None,
        help="Subdir where to save the evaluator config and results.",
    )
    p.add_argument(
        "--load-model-path",
        type=Path,
        default=None,
        help="Path to load the model from.",
    )
    p.add_argument(
        "--arch",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        default="vitl14",
        help="DINOv2 backbone variant",
    )
    p.add_argument(
        "--layers",
        choices=[1, 4],
        type=int,
        default=4,
        help="How many intermediate layers feed the linear head.",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size for evaluation."
    )
    p.add_argument(
        "--num-workers", type=int, default=4, help="PyTorch DataLoader workers."
    )
    p.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 5],
        help="Compute accuracy for these k values.",
    )
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    p.add_argument(
        "--reg", action="store_true", help="Load register-token variant (_reg)"
    )
    p.add_argument("--api", action="store_true", help="Hide model as if it was an API.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---------- add outfile to outdir ----------
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    if args.out_name is None:
        args.out_name = timestamp
    args.out_dir = os.path.join(args.out_dir, args.out_name)

    # ---------- dataset & loader ----------
    val_loader = HDF5Dataset.get_dataloader(
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        seed=42,
        rank=0,
        world_size=1,
        **dataset_configs[args.dataset_config],
    )

    # ---------- model ----------
    suffix = "_reg_lc" if args.reg else "_lc"
    hub_entry = f"dinov2_{args.arch}{suffix}"
    backbone = torch.hub.load(
        repo_or_dir="./dinov2",
        model=hub_entry,
        source="local",
        pretrained=True,
        layers=args.layers,
    )

    optim_f = optimizer_configs[args.optim_config]["factory"](
        **optimizer_configs[args.optim_config]["config"]
    )
    model = tta_configs[args.tta_config]["factory"](
        model=backbone,
        optimizer_f=optim_f,
        **tta_configs[args.tta_config]["config"],
    )

    # ---------- evaluation ----------
    evaluator = ClfEvaluator(
        model=model,
        topk=args.topk,
        device=args.device,
        load_model_path=args.load_model_path,
        save_dir=Path(args.out_dir),
        save_model=True,
        enable_grad=True,
    )
    evaluator.save_config(
        "config.json",
        extra_config={
            "tta_config": tta_configs[args.tta_config],
            "optimizer_config": optimizer_configs[args.optim_config],
            "dataset_config": dataset_configs[args.dataset_config],
            "reg": args.reg,
            "api": args.api,
        },
    )
    results = evaluator.evaluate(val_loader)
    evaluator.save_results(
        results, f"{args.dataset_config.replace('/', '-')}_results.json"
    )
    print("Finished. Accuracies:", results)


if __name__ == "__main__":
    main()
