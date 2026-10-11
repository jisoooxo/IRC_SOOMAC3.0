import os


def configure_torch_memory() -> None:
    """Apply the server-provided PyTorch allocator fraction before model allocation."""
    value = os.environ.get("GPU_MEMORY_FRACTION")
    if value is None:
        return

    import torch

    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError("GPU_MEMORY_FRACTION must be greater than 0 and at most 1")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)

