"""
Frequency Attention Module (FAM) for Medical Ultrasound Segmentation
Designed for multi-class segmentation with blurred boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyAttentionModule(nn.Module):
    """
    Frequency Attention Module using 2D FFT for extracting multi-frequency features.
    Particularly effective for medical ultrasound images with blurred boundaries.

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for efficiency
        cutoff: Frequency cutoff ratio for separating low/mid/high frequencies
    """

    def __init__(self, in_channels, reduction=4, cutoff=0.3):
        super(FrequencyAttentionModule, self).__init__()

        self.in_channels = in_channels
        self.reduction = reduction
        # cutoff as learnable parameter, initialized from provided value
        self.cutoff = nn.Parameter(torch.tensor(cutoff, dtype=torch.float32))

        mid_channels = in_channels // reduction

        # Shared MLP for channel attention generation from frequency features
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Output projection with residual connection
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Enhanced features with frequency attention [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Extract multi-frequency features
        low_freq, mid_freq, high_freq = self.filter_frequency_bands(x)

        # Concatenate frequency features along channel dimension
        freq_features = torch.cat([low_freq, mid_freq, high_freq], dim=1)  # [B, C*3, H, W]

        # Generate channel-wise attention weights from frequency features
        attention = self.mlp(freq_features)  # [B, C, H, W]

        # Apply attention to input features
        attended = x * attention

        # Output projection with residual connection
        out = self.output_conv(attended)
        out = self.gamma * out + x

        return out

    def filter_frequency_bands(self, tensor):
        """
        Apply 2D FFT and separate into low, mid, high frequency bands.

        Args:
            tensor: Input tensor [B, C, H, W]
        Returns:
            Tuple of (low_freq, mid_freq, high_freq) tensors
        """
        B, C, H, W = tensor.shape
        device = tensor.device

        # Ensure float type for FFT
        tensor = tensor.float()

        # Compute 2D FFT and shift zero frequency to center
        fft_tensor = torch.fft.fftshift(
            torch.fft.fft2(tensor, dim=(-2, -1)),
            dim=(-2, -1)
        )

        # Calculate distance from center for each position
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, device=device).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, device=device).view(1, W).expand(H, W)
        distance = torch.sqrt((y_coords - center_y).float() ** 2 +
                             (x_coords - center_x).float() ** 2)

        # Calculate max radius and cutoffs
        max_radius = torch.sqrt(torch.tensor(center_y ** 2 + center_x ** 2, device=device).float())
        # Use sigmoid to constrain cutoff to (0, 1) range
        cutoff_val = torch.sigmoid(self.cutoff)
        low_cutoff = max_radius * cutoff_val
        high_cutoff = max_radius * (1 - cutoff_val)

        # Create frequency band masks
        low_mask = (distance <= low_cutoff).float()
        high_mask = (distance >= high_cutoff).float()
        mid_mask = ((distance > low_cutoff) & (distance < high_cutoff)).float()

        # Expand masks to match batch and channel dimensions
        low_mask = low_mask.view(1, 1, H, W)
        high_mask = high_mask.view(1, 1, H, W)
        mid_mask = mid_mask.view(1, 1, H, W)

        # Apply masks to frequency domain
        low_freq_fft = fft_tensor * low_mask
        mid_freq_fft = fft_tensor * mid_mask
        high_freq_fft = fft_tensor * high_mask

        # Inverse FFT to get spatial domain features
        low_freq = torch.fft.ifft2(
            torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        mid_freq = torch.fft.ifft2(
            torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        high_freq = torch.fft.ifft2(
            torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        return low_freq, mid_freq, high_freq


class FAMBlock(nn.Module):
    """
    Frequency Attention Block combining FAM with residual connection.
    Suitable for integration into encoder/decoder networks.
    """

    def __init__(self, in_channels, reduction=4, cutoff=0.3):
        super(FAMBlock, self).__init__()

        self.fam = FrequencyAttentionModule(in_channels, reduction, cutoff)

        # Additional refinement convolution
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Refined features [B, C, H, W]
        """
        out = self.fam(x)
        out = self.refine(out)
        return out


class MultiScaleFAM(nn.Module):
    """
    Multi-scale Frequency Attention Module for capturing features at different scales.
    Particularly useful for medical images with varying lesion sizes.
    """

    def __init__(self, in_channels, scales=[0.2, 0.35, 0.5], reduction=4):
        super(MultiScaleFAM, self).__init__()

        self.scales = scales
        self.fam_modules = nn.ModuleList([
            FrequencyAttentionModule(in_channels, reduction, cutoff=scale)
            for scale in scales
        ])

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Multi-scale fused features [B, C, H, W]
        """
        multi_scale_features = []

        for fam in self.fam_modules:
            feat = fam(x)
            multi_scale_features.append(feat)

        # Concatenate and fuse
        concat_features = torch.cat(multi_scale_features, dim=1)
        out = self.fusion(concat_features)

        return out


# Standalone test
if __name__ == '__main__':
    # Test FAM module
    batch_size = 4
    channels = 256
    height, width = 60, 80

    x = torch.rand(batch_size, channels, height, width)

    # Test basic FAM
    fam = FrequencyAttentionModule(in_channels=channels)
    out = fam(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test FAMBlock
    fam_block = FAMBlock(in_channels=channels)
    out_block = fam_block(x)
    print(f"FAMBlock output shape: {out_block.shape}")

    # Test MultiScaleFAM
    multi_fam = MultiScaleFAM(in_channels=channels)
    out_multi = multi_fam(x)
    print(f"MultiScaleFAM output shape: {out_multi.shape}")

    # Verify output dimensions match input
    assert out.shape == x.shape, "Output shape mismatch!"
    assert out_block.shape == x.shape, "FAMBlock output shape mismatch!"
    assert out_multi.shape == x.shape, "MultiScaleFAM output shape mismatch!"

    print("\nAll tests passed!")
