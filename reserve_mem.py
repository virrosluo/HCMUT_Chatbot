import torch

# Reserve 8GB VRAM
a = torch.rand((2, 1024, 1024, 1024), device="cuda")
breakpoint()
