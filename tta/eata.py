"""
Refactored EATA in the style of EnergyTTA:
- episodic reset using saved (model, optimizer) states
- per-step adaptation via `forward_and_adapt`
- internal counters and moving-average probs handled as members
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable

from .utils import load_model_and_optimizer, copy_model_and_optimizer, softmax_entropy


class EATA(nn.Module):
    """
    Entropy-based sample selection with redundancy filtering (EATA).
    Structured like EnergyTTA:
      - saved (model/optimizer) state and episodic reset
      - step loop in forward()
      - per-step adaptation in forward_and_adapt()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_f: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        steps: int = 1,
        fishers: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        fisher_alpha: float = 2000.0,
        episodic: bool = False,
        e_margin: float = math.log(1000) / 2 - 1,  # E_0 (Eq. 3)
        d_margin: float = 0.05,  # epsilon threshold (Eq. 5)
        eval_mode_during_tta: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"

        self.model = model
        self.optimizer_f = optimizer_f
        self.steps = steps
        self.episodic = episodic
        self.eval_mode_during_tta = eval_mode_during_tta

        # Sample-count trackers
        self.num_samples_update_1 = 0  # reliable (after first filter)
        self.num_samples_update_2 = 0  # reliable & non-redundant (after second filter)

        # Hyperparameters
        self.e_margin = e_margin
        self.d_margin = d_margin

        # Moving average of probability vector (Eq. 4)
        self.current_model_probs: torch.Tensor | None = None

        # EWC-style fisher regularizer (Eq. 9), weighted in Eq. 8
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha

        # keep reset snapshots
        self.optimizer = self.optimizer_f(self.model)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def get_base_model(self):
        return self.model

    @staticmethod
    def _update_model_probs(
        current_model_probs: torch.Tensor | None,
        new_probs: torch.Tensor | None,
        momentum: float = 0.9,
    ) -> torch.Tensor | None:
        """EMA of class-probability vectors (Eq. 4 in EATA)."""
        if new_probs is None or new_probs.size(0) == 0:
            return current_model_probs
        with torch.no_grad():
            new_mean = new_probs.mean(0)
            if current_model_probs is None:
                return new_mean
            return momentum * current_model_probs + (1.0 - momentum) * new_mean

    # ----------------------------- lifecycle ----------------------------- #
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    def reset_steps(self, new_steps: int):
        assert new_steps > 0
        self.steps = new_steps

    def reset_model_probs(self, probs: torch.Tensor | None):
        self.current_model_probs = probs

    # ------------------------- adaptation step -------------------------- #
    @torch.enable_grad()  # ensure grads even if called under no_grad
    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        One EATA update step:
          1) forward
          2) compute entropy, filter unreliable (E < E_0)
          3) redundancy filter via cosine sim. to EMA probs
          4) reweight entropy and add EWC if provided
          5) update (only if selected batch is non-empty)
        Returns model outputs on the (unmodified) input x.
        """
        was_training = self.model.training
        if self.eval_mode_during_tta:
            self.model.eval()

        outputs = self.model(x)  # logits, shape (B, C)

        # --- entropy + first filter (reliable)
        ent = softmax_entropy(outputs)  # shape (B,)
        filter_ids_1 = torch.where(ent < self.e_margin)  # tuple with 1D idx
        ids1 = filter_ids_1

        # selected entropies / probs for subsequent ops
        sel_ent = ent[filter_ids_1]
        sel_probs = outputs[filter_ids_1].softmax(1)

        # --- second filter (redundancy) using cosine similarity to EMA probs
        ids2 = torch.where(ids1[0] > -0.1)  # default: pass-through
        updated_probs = None

        if self.current_model_probs is not None and sel_probs.size(0) > 0:
            cos_sim = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0), sel_probs, dim=1
            )
            filter_ids_2 = torch.where(torch.abs(cos_sim) < self.d_margin)
            ids2 = filter_ids_2
            sel_ent = sel_ent[filter_ids_2]
            sel_probs = sel_probs[filter_ids_2]
            updated_probs = self._update_model_probs(
                self.current_model_probs, sel_probs
            )
        else:
            updated_probs = self._update_model_probs(
                self.current_model_probs, sel_probs
            )

        # reweight entropy (inverse temperature based on margin distance)
        if sel_ent.numel() > 0:
            coeff = 1.0 / torch.exp(sel_ent.detach().clone() - self.e_margin)
            loss = (sel_ent * coeff).mean(0)
        else:
            loss = torch.zeros((), device=outputs.device, dtype=outputs.dtype)

        # EWC (anti-forgetting) regularizer
        if self.fishers is not None:
            ewc_loss = torch.zeros((), device=outputs.device, dtype=outputs.dtype)
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    fisher_F, theta_old = self.fishers[name]
                    ewc_loss = (
                        ewc_loss
                        + self.fisher_alpha
                        * (fisher_F * (param - theta_old) ** 2).sum()
                    )
            loss = loss + ewc_loss

        # update only if we actually selected some samples
        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # bookkeeping (match original counters) + EMA probs
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += sel_ent.size(0)
        self.reset_model_probs(updated_probs)

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
