import torch


def get_all_params(model: torch.nn.Module):
    return model.parameters()


def get_norm_params(model: torch.nn.Module):
    ln_params = []
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.LayerNorm,
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ),
        ):
            ln_params.extend(list(module.parameters()))

    return ln_params


def optim_f(
    get_params_f=get_all_params,
    optim=torch.optim.SGD,
    optim_config={
        "lr": 2.5e-4,
        "momentum": 0.9,
        "weight_decay": 0.0,
    },
):
    """
    Create an optimizer for an API wrapped model.
    """

    return lambda model: optim(get_params_f(model), **optim_config)


optimizer_configs = {
    "default": {
        "factory": optim_f,
        "config": {
            "get_params_f": get_all_params,
            "optim": torch.optim.SGD,
            "optim_config": {
                "lr": 2.5e-4,
                "momentum": 0.9,
                "weight_decay": 0.0,
            },
        },
    },
    "norm": {
        "factory": optim_f,
        "config": {
            "get_params_f": get_norm_params,
            "optim": torch.optim.SGD,
            "optim_config": {
                "lr": 2.5e-4,
                "momentum": 0.9,
                "weight_decay": 0.0,
            },
        },
    },
}
