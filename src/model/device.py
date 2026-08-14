"""Which device tensors live on.

Previously `GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else
"cpu")` appeared verbatim at the top of both `network.py` and `train.py`. That
had two costs:

  * `torch.cuda.is_available()` ran at import. Importing anything under
    `src.model` -- including from a test that never touches a GPU -- paid for
    CUDA driver initialisation, and the answer was then frozen for the life of
    the process.

  * Two independent copies. Nothing kept them agreeing, and nothing would have
    reported it if they had diverged.

Resolved on first call and cached, so the probe happens when a tensor actually
needs placing.
"""

import torch

_DEVICE = None


def gpu_device() -> torch.device:
    """The device to place tensors on, probed once and cached."""

    global _DEVICE

    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return _DEVICE


def set_device(device) -> None:
    """Override the cached device.

    Exists for tests that need to pin execution to the CPU regardless of what
    the host has; there is no production caller.
    """

    global _DEVICE

    _DEVICE = torch.device(device) if device is not None else None
