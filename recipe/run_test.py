import torch
from piqa import PSNR, SSIM


image = torch.rand(1, 3, 16, 16)
perturbed = torch.clamp(image * 0.9 + 0.05, 0.0, 1.0)

psnr = PSNR()(image, perturbed)
ssim = SSIM(window_size=7, n_channels=3)(image, perturbed)

assert torch.isfinite(psnr)
assert psnr.item() > 20.0
assert 0.0 < ssim.item() <= 1.0
