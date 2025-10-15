from copy import deepcopy
from .utils import (
    init_random,
    copy_model_and_optimizer,
    load_model_and_optimizer,
)
from torchvision.utils import save_image
import os
import torch
import torch.nn as nn
import numpy as np


class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        logits = self.f(x)
        return logits

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits

    def sample_p_0(self, reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
        if len(replay_buffer) == 0:
            return init_random(bs, im_sz=im_sz, n_ch=n_ch), []
        buffer_size = len(replay_buffer)
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds

        buffer_samples = replay_buffer[inds]
        random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)
        choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(
        self,
        replay_buffer,
        n_steps,
        sgld_lr,
        sgld_std,
        reinit_freq,
        batch_size,
        im_sz,
        n_ch,
        device,
        y=None,
    ):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """

        # get batch size
        bs = batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = self.sample_p_0(
            reinit_freq=reinit_freq,
            replay_buffer=replay_buffer,
            bs=bs,
            im_sz=im_sz,
            n_ch=n_ch,
            device=device,
            y=y,
        )
        init_samples = deepcopy(init_sample)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(
                self(x_k, y=y)[0].sum(), [x_k], retain_graph=True
            )[0]
            x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)

        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples, init_samples.detach()


class EnergyTTA(nn.Module):
    def __init__(
        self,
        model,
        optimizer_f,
        steps=1,
        episodic=False,
        buffer_size=10000,
        sgld_steps=20,
        sgld_lr=1,
        sgld_std=0.01,
        reinit_freq=0.05,
        if_cond=False,
        n_classes=10,
        im_sz=32,
        n_ch=3,
        path=None,
        eval_mode_during_sgld: bool = True,
        eval_mode_during_tta: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.energy_model = EnergyModel(model)
        self.replay_buffer = init_random(buffer_size, im_sz=im_sz, n_ch=n_ch)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer_f = optimizer_f
        self.steps = steps
        assert steps > 0, "tea requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond

        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch

        self.path = path

        self.eval_mode_during_sgld = eval_mode_during_sgld
        self.eval_mode_during_tta = eval_mode_during_tta

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.optimizer = self.optimizer_f(self.energy_model)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.energy_model, self.optimizer
        )

    def get_base_model(self):
        return self.energy_model

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.energy_model, self.optimizer, self.model_state, self.optimizer_state
        )

    @torch.enable_grad()
    def visualize_images(
        self,
        batch_size: int,
        device: torch.device | None = None,
        counter: int | None = None,
        step: int | None = None,
    ):
        repeat_times = batch_size // self.n_classes
        y = torch.arange(self.n_classes).repeat(repeat_times).to(device)

        x_fake, _ = self.energy_model.sample_q(
            self.replay_buffer,
            n_steps=self.sgld_steps,
            sgld_lr=self.sgld_lr,
            sgld_std=self.sgld_std,
            reinit_freq=self.reinit_freq,
            batch_size=batch_size,
            im_sz=self.im_sz,
            n_ch=self.n_ch,
            device=device,
            y=y,
        )
        save_image(
            x_fake.detach().cpu(),
            os.path.join(self.path, "sample.png"),
            padding=2,
            nrow=np.ceil(np.sqrt(batch_size)).astype(int),
        )

        buffer_new = self.replay_buffer.cpu()
        buffer_old = self.replay_buffer_old.cpu()
        buffer_diff = buffer_new - buffer_old

        if step == 0:
            save_image(
                buffer_old,
                os.path.join(self.path, "buffer_init.png"),
                padding=2,
                nrow=np.ceil(np.sqrt(len(buffer_old))).astype(int),
            )

        tag = f"buffer-{counter}-{step}"
        save_image(
            buffer_new,
            os.path.join(self.path, f"{tag}.png"),
            padding=2,
            nrow=np.ceil(np.sqrt(len(buffer_new))).astype(int),
        )
        save_image(
            buffer_diff,
            os.path.join(self.path, "buffer_diff.png"),
            padding=2,
            nrow=np.ceil(np.sqrt(len(buffer_diff))).astype(int),
        )

    @torch.enable_grad()  # ensure grads even in eval mode
    def forward_and_adapt(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Perform a SINGLE update step of EnergyTTA:
        - Generate x_fake via SGLD from replay buffer (cond/uncond)
        - Compute energy difference on real vs. fake
        - Minimize -(E_real - E_fake) w.r.t. chosen params

        Mode toggling for SGLD and TTA happens here and is restored after the step.
        """
        # Basic shapes / device
        B = x.shape[0]
        C = x.shape[1]
        im_sz = x.shape[2]  # assumes square inputs
        device = x.device

        # Remember original mode to restore later
        was_training = self.energy_model.training

        # ------------------ Phase 1: SGLD sampling mode ------------------ #
        # If eval_mode_during_sgld == True → we want eval(); else train()
        desired_training_sgld = not self.eval_mode_during_sgld
        if self.energy_model.training != desired_training_sgld:
            self.energy_model.train(desired_training_sgld)

        # Class-conditional vs unconditional sampling
        if self.if_cond in ("uncond", False):
            y = None
        elif self.if_cond in ("cond", True):
            y = torch.randint(self.n_classes, (B,), device=device)
        else:
            raise ValueError("`if_cond` must be 'cond', 'uncond', True, or False.")

        x_fake, _ = self.energy_model.sample_q(
            replay_buffer=self.replay_buffer,
            n_steps=self.sgld_steps,
            sgld_lr=self.sgld_lr,
            sgld_std=self.sgld_std,
            reinit_freq=self.reinit_freq,
            batch_size=B,
            im_sz=im_sz,
            n_ch=C,
            device=device,
            y=y,
        )

        # ------------------ Phase 2: TTA update mode --------------------- #
        # If eval_mode_during_tta == True → eval(); else train()
        desired_training_tta = not self.eval_mode_during_tta
        if self.energy_model.training != desired_training_tta:
            self.energy_model.train(desired_training_tta)

        # Compute energies and loss
        energy_real = self.energy_model(x)[0].mean()
        energy_fake = self.energy_model(x_fake)[0].mean()
        loss = -(energy_real - energy_fake)

        # Update
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Restore original mode
        self.energy_model.train(was_training)

    def forward(
        self,
        x,
        if_adapt=True,
        counter=None,
        if_vis=False,
    ):
        if self.episodic:
            self.reset()

        if if_adapt:
            for i in range(self.steps):
                # single adaptation step
                self.forward_and_adapt(x)

                # Optional visualization (kept in forward to preserve original timing)
                if if_vis:
                    self.visualize_images(
                        batch_size=100,
                        device=x.device,
                        counter=counter,
                        step=i,
                    )

        # Deterministic final prediction
        was_training = self.energy_model.training
        self.energy_model.eval()
        with torch.no_grad():
            outputs = self.energy_model.classify(x)
        self.energy_model.train(was_training)

        return outputs
