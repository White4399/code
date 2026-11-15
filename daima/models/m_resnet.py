"""
ResNet50 Encoder for few shot segmentation with FPN support
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    """
    ResNet50 Encoder with multi-scale feature extraction for FPN
    
    Returns features at 4 stages:
        - stage1: after first maxpool (1/4 resolution)
        - stage2: after layer1 (1/4 resolution)
        - stage3: after layer2 (1/8 resolution)
        - stage4: after layer3 (1/16 resolution)
    """
    
    def __init__(self, pretrained_path=None, load_pretrained=True):
        super(ResNetEncoder, self).__init__()

        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64, 3, stride=1, dilation=1)   # 1/4 resolution
        self.layer2 = self._make_layer(128, 4, stride=2, dilation=1)  # 1/8 resolution
        self.layer3 = self._make_layer(256, 6, stride=2, dilation=1)  # 1/16 resolution
        self.layer4 = self._make_layer(512, 3, stride=1, dilation=2)  # 1/16 resolution with dilation

        self._init_weights()

        if load_pretrained:
            self._load_pretrained(pretrained_path)
    
    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * Bottleneck.expansion
        
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self, pretrained_path):
        """Load ImageNet ResNet50 weights from a file and/or torchvision (shape-safe merge)."""
        dic = None
        source_msg = 'torchvision'

        if pretrained_path is not None:
            p = str(pretrained_path).strip()
            if p and os.path.isfile(p):
                try:
                    dic = torch.load(p, map_location='cpu')
                    source_msg = p
                except Exception as e:
                    print(f"Warning: failed to read {p}: {e}")

        if dic is None:
            try:
                from torchvision.models import resnet50, ResNet50_Weights

                dic = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
            except Exception:
                try:
                    from torchvision.models import resnet50

                    dic = resnet50(pretrained=True).state_dict()
                except Exception as e:
                    print(f"Warning: could not load torchvision ResNet50: {e}")
                    return

        if isinstance(dic, dict):
            if 'state_dict' in dic and isinstance(dic['state_dict'], dict):
                dic = dic['state_dict']
            elif 'model' in dic and isinstance(dic['model'], dict):
                dic = dic['model']

        if any(isinstance(k, str) and k.startswith('module.') for k in dic.keys()):
            dic = {k.replace('module.', ''): v for k, v in dic.items()}

        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in dic.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Merged {len(pretrained_dict)}/{len(model_dict)} tensors from {source_msg}")
    
    def forward(self, x):
        """Return multi-scale features for FPN"""
        features = []
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stage 1 (1/4 resolution)
        x = self.layer1(x)
        features.append(x)  # C4
        
        # Stage 2 (1/8 resolution)
        x = self.layer2(x)
        features.append(x)  # C3
        
        # Stage 3 (1/16 resolution)
        x = self.layer3(x)
        features.append(x)  # C2
        
        # Stage 4 (1/16 resolution, dilated)
        x = self.layer4(x)
        features.append(x)  # C1
        
        return features