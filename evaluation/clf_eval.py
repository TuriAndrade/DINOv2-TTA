from pathlib import Path
from typing import Sequence, Optional, Dict
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import CustomJSONEncoder


class ClfEvaluator:
    """
    Minimal evaluator for classifiers on any DataLoader returning (image, label).

    Parameters
    ----------
    model : nn.Module
        An initialized model (already wrapped with any TTA if desired).
    save_dir : Path
        Directory to save results/config and (optionally) model state.
    topk : Sequence[int]
        Accuracy@k to compute; default (1,) → top-1 only.
    device : {"cuda","cpu"}
        Device to run on.
    load_model_path : Optional[Path]
        If provided, loads the full model state_dict before evaluation.
    save_model : bool
        If True, saves model state_dict after evaluation to `save_dir / <filename>`.
    enable_grad : bool
        If True, enable gradients during evaluate() (useful for TTA); otherwise uses no_grad().
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: Path,
        topk: Sequence[int] = (1,),
        device: str = "cuda",
        load_model_path: Optional[Path] = None,
        save_model: bool = False,
        enable_grad: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model: nn.Module = model  # keep on CPU until (optional) load
        if load_model_path is not None:
            self.load_model(load_model_path)

        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()

        self.topk = tuple(topk)
        self.save_model_flag = save_model
        self.enable_grad = enable_grad

        self.params_to_save = [
            "model",
            "topk",
            "device",
            "enable_grad",
        ]

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Returns dict like {'top1': 76.7, 'top5': 93.2} (percent),
        showing a running tqdm progress-bar.

        Gradients are enabled or disabled according to `self.enable_grad`.
        """
        correct = {k: 0 for k in self.topk}
        total = 0

        ctx = torch.enable_grad() if self.enable_grad else torch.no_grad()
        with ctx:
            for imgs, labels in tqdm(loader, desc="Evaluating", unit="batch"):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(imgs)  # (B, C)
                total += labels.size(0)

                for k in self.topk:
                    preds = logits.topk(k, 1).indices  # (B, k)
                    correct[k] += (
                        preds.eq(labels.unsqueeze(1)).any(dim=1).float().sum().item()
                    )

        results = {
            f"top{k}": (100.0 * correct[k] / total) if total > 0 else 0.0
            for k in self.topk
        }

        if self.save_model_flag:
            self.save_model("model_state.pt")

        return results

    def save_results(self, results: Dict[str, float], filename: str = "results.json"):
        path = self.save_dir / filename
        path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {path.resolve()}")

    def save_config(self, filename: str = "config.json"):
        cfg = {
            "model_class": self.model.__class__.__name__,
            "topk": self.topk,
            "device": str(self.device),
            "enable_grad": self.enable_grad,
        }
        path = self.save_dir / filename
        path.write_text(json.dumps(cfg, indent=2, cls=CustomJSONEncoder))
        print(f"Config saved to {path.resolve()}")

    # ----------------------------- Save / Load ------------------------------ #
    def save_model(self, filename: str = "model_state.pt"):
        """Save full model state_dict to `save_dir / filename`."""
        path = self.save_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model state saved to {path.resolve()}")

    def load_model(self, path: Path, strict: bool = True):
        """Load full model state_dict from `path` into the current model."""
        path = Path(path)
        if not path.exists():
            print(f"[ClfEvaluator] Model state file not found: {path}")
            return
        try:
            state = torch.load(path, map_location="cpu")
            if not isinstance(state, dict):
                print(
                    f"[ClfEvaluator] Expected a state_dict (dict) in {path.name}, got {type(state)}"
                )
                return
            self.model.load_state_dict(state, strict=strict)
            print(f"Loaded full model state from {path.resolve()}")
        except Exception as e:
            print(f"[ClfEvaluator] Failed to load model state: {e}")

    def save_config(self, filename: str = "config.json", extra_config: dict = None):
        config = {k: getattr(self, k) for k in self.params_to_save}

        if extra_config is not None:
            config = {**config, **extra_config}

        path = self.save_dir / filename
        path.write_text(json.dumps(config, indent=2, cls=CustomJSONEncoder))
        print(f"Config saved to {path.resolve()}")
