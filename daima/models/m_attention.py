"""
Multi-Head Self-Attention Module for Medical Ultrasound Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module
    
    Args:
        in_channels: input channels
        num_heads: number of attention heads
        key_channels: channels for key and query
        value_channels: channels for value
    """
    
    def __init__(self, in_channels, num_heads=8, key_channels=None, value_channels=None):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.key_channels = key_channels or in_channels // num_heads
        self.value_channels = value_channels or in_channels // num_heads
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, self.key_channels * num_heads, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.key_channels * num_heads, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.value_channels * num_heads, kernel_size=1)
        
        # Output projection
        self.out_conv = nn.Conv2d(self.value_channels * num_heads, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        # Project to Q, K, V
        query = self.query_conv(x).view(batch_size, self.num_heads, self.key_channels, H * W)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.key_channels, H * W)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.value_channels, H * W)
        
        # Attention
        attention = torch.matmul(query.transpose(-2, -1), key)  # B x H x (H*W) x (H*W)
        attention = attention / math.sqrt(self.key_channels)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(value, attention.transpose(-2, -1))  # B x H x Cv x (H*W)
        out = out.view(batch_size, self.num_heads * self.value_channels, H, W)
        
        # Output projection with residual connection
        out = self.out_conv(out)
        out = self.gamma * out + x
        
        return out


class ChannelAttention(nn.Module):
    """Channel Attention Module for suppressing noise"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x + x  # Residual connection


class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on lesion regions"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out) * x + x


class MixedAttention(nn.Module):
    """
    Mixed Attention Module combining Channel and Spatial attention
    """
    
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(MixedAttention, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x