"""
Combined Loss Functions for Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for medical image segmentation"""
    
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # pred: [N, C, H, W] logits
        # target: [N, H, W] labels
        
        pred = F.softmax(pred, dim=1)
        
        # 创建有效区域 mask
        valid_mask = target != self.ignore_index
        target = target.clone()
        target[~valid_mask] = 0
        
        n_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()
        
        # 应用有效区域 mask
        valid_mask = valid_mask.unsqueeze(1).float()
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return (1 - dice.mean())


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = torch.full_like(target, self.alpha, dtype=torch.float)
            alpha_t[target == 0] = 1 - self.alpha
            alpha_t[target == self.ignore_index] = 0
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Focal Loss"""
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal