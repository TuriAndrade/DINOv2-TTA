import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable

from .utils import load_model_and_optimizer, copy_model_and_optimizer, softmax_entropy


class Tent(nn.Module):
    """
    Tent adapts a model by entropy minimization during testing.

    Structured like EnergyTTA/EATA:
      - episodic reset
      - per-step adaptation in forward_and_adapt()
      - step loop inside forward()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_f: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        steps: int = 1,
        episodic: bool = False,
        eval_mode_during_tta: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert steps > 0, "Tent requires >= 1 step(s) to forward and update"
        self.model = model
        self.optimizer_f = optimizer_f
        self.steps = steps
        self.episodic = episodic
        self.eval_mode_during_tta = eval_mode_during_tta

        # save states for episodic reset
        self.optimizer = self.optimizer_f(self.model)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def get_base_model(self):
        return self.model

    # ----------------------------- lifecycle ----------------------------- #
    def reset(self):
        """Reset model and optimizer to their saved initial states."""
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    def reset_steps(self, new_steps: int):
        assert new_steps > 0
        self.steps = new_steps

    # ------------------------- adaptation step -------------------------- #
    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor) -> None:
        """
        Perform a SINGLE Tent update in-place for batch x.
        Mode toggling (eval during TTA if configured) is handled here and restored.
        """
        was_training = self.model.training
        if self.eval_mode_during_tta:
            self.model.eval()

        outputs = self.model(x)
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # restore original mode
        self.model.train(was_training)

    # ------------------------------ forward ----------------------------- #
    def forward(self, x: torch.Tensor, if_adapt: bool = True) -> torch.Tensor:
        if self.episodic:
            self.reset()

        if if_adapt:
            for _ in range(self.steps):
                self.forward_and_adapt(x)

        # deterministic final prediction; restore mode after
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        self.model.train(was_training)

        return outputs
