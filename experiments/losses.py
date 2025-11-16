import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gaussian_kernel(window_size, sigma):
    gauss = torch.tensor(
        [math.exp(-(x - window_size//2)**2 / float(2 * sigma**2))
         for x in range(window_size)]
    )
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(pred, target, window_size=11, data_range=1.0):
    """
    pred, target: NCHW tensors in [0,1]
    """
    (_, channel, _, _) = pred.size()
    window = create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

class L1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.5, window_size=11):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_value = ssim(pred, target, window_size=self.window_size)
        ssim_loss = 1 - ssim_value
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
