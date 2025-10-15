import torch
from copy import deepcopy


@torch.jit.script
def softmax_entropy(
    x: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x = x / temperature
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def init_random(bs, im_sz=32, n_ch=3):
    return torch.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
