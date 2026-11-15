"""
Fewshot Semantic Segmentation with ResNet50 + FPN + MixedAttention + PPM
Based on PSPNet's Pyramid Pooling Module for enhanced spatial context.

Ablation-aware implementation: every optional module (FPN, PPM, Cross-Attn)
can be toggled on/off via cfg['ablation'] without changing downstream dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .m_resnet import ResNetEncoder
from .m_attention import MixedAttention


class LightweightSpatialChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_weight = self.channel_attn(x)
        x = x * channel_weight
        spatial_input = torch.cat(
            [x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values],
            dim=1,
        )
        spatial_weight = self.spatial_attn(spatial_input)
        return x * spatial_weight


class SupportQueryCrossAttention(nn.Module):
    def __init__(self, channels, reduction=16, temperature=1.0):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.support_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.temperature = temperature

    def forward(self, support_feat, query_feat):
        support_proj = self.support_proj(support_feat)
        query_proj = self.query_proj(query_feat)
        value = self.value_proj(support_feat)

        attn_map = (query_proj * support_proj).sum(dim=1, keepdim=True) / max(self.temperature, 1e-6)
        attn_map = torch.softmax(attn_map.flatten(2), dim=-1).view_as(attn_map)

        support_context = value * attn_map
        support_context = support_context.sum(dim=(2, 3), keepdim=True)
        support_context = support_context.expand_as(query_feat)

        fused = self.fuse(torch.cat([query_feat, support_context], dim=1))
        gate = self.refine(fused)
        return query_feat + fused * gate


class LightweightBidirectionalSupportQueryAttention(nn.Module):
    def __init__(self, channels, reduction=16, temperature=1.0, feedback_ratio=0.5):
        super().__init__()
        self.support_to_query = SupportQueryCrossAttention(
            channels, reduction=reduction, temperature=temperature
        )
        self.query_to_support = SupportQueryCrossAttention(
            channels, reduction=reduction, temperature=temperature
        )
        self.feedback_ratio = feedback_ratio
        self.query_feedback_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.support_feedback_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, support_feat, query_feat):
        query_from_support = self.support_to_query(support_feat, query_feat)
        support_from_query = self.query_to_support(query_feat, support_feat)

        query_update = self.query_feedback_proj(query_from_support - query_feat)
        support_update = self.support_feedback_proj(support_from_query - support_feat)

        refined_query = query_feat + query_update
        refined_support = support_feat + self.feedback_ratio * support_update
        return refined_support, refined_query


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) from PSPNet.
    Extracts multi-scale spatial context using different pooling scales.

    Args:
        in_channels: Number of input channels
        pool_scales: List of pooling scales (default: [1, 2, 3, 6])
        reduction: Channel reduction ratio
    """

    def __init__(self, in_channels, pool_scales=(1, 2, 3, 6), reduction=4):
        super(PyramidPoolingModule, self).__init__()

        self.pool_scales = pool_scales
        self.in_channels = in_channels
        mid_channels = in_channels // reduction

        # Each pooling scale has its own conv to reduce channels
        self.pool_convs = nn.ModuleList()
        for scale in pool_scales:
            self.pool_convs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale) if scale > 1 else nn.Identity(),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            ))

        # Final conv to fuse pooled features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels * len(pool_scales), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Enhanced features with multi-scale spatial context [B, C, H, W]
        """
        input_size = x.size()[2:]

        pooled_features = [x]  # Start with original features

        for pool_conv in self.pool_convs:
            # Apply pooling at this scale
            pooled = pool_conv(x)
            # Upsample back to original size
            pooled = F.interpolate(pooled, size=input_size, mode='bilinear', align_corners=True)
            pooled_features.append(pooled)

        # Concatenate all scales along channel dimension
        output = torch.cat(pooled_features, dim=1)

        # Fuse features
        output = self.fusion_conv(output)

        return output


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    """

    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()

        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Top-down path
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Attention module for feature refinement
        self.attention = MixedAttention(out_channels)

    def forward(self, features):
        """
        Args:
            features: list of features from ResNet stages [C4, C3, C2, C1]
        Returns:
            fused multi-scale features
        """
        # Lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            upsample = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsample

        # Output convolutions
        outputs = []
        for i, lateral in enumerate(laterals):
            out = self.output_convs[i](lateral)
            outputs.append(out)

        # Use the finest level as main feature
        main_feat = outputs[0]

        # Apply attention refinement
        main_feat = self.attention(main_feat)

        return main_feat


class FewShotSegResNet(nn.Module):
    """
    Fewshot Segmentation model with ResNet50 backbone, FPN, MixedAttention and PPM.

    Architecture:
        ResNet50 -> FPN -> MixedAttention -> PPM -> Output Projection -> Prototype Matching

    Ablation support (controlled via cfg['ablation']):
        cfg['ablation']['fpn']        : bool — enable FPN (otherwise channel-aligning conv)
        cfg['ablation']['ppm']        : bool — enable Pyramid Pooling Module
        cfg['ablation']['cross_attn'] : bool — enable bidirectional support-query cross attention

    All branches output the same channel dimension (256) so the rest of the pipeline
    is agnostic to which ablation variant is in use.
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, load_encoder_pretrained=True):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.ablation = (self.config.get('ablation')
                         if isinstance(self.config, dict) and 'ablation' in self.config
                         else {})
        # Default: every module ON
        self.use_fpn = bool(self.ablation.get('fpn', True))
        self.use_ppm = bool(self.ablation.get('ppm', True))
        self.use_cross_attn = bool(self.ablation.get('cross_attn', True))

        self.out_channels = 256  # canonical feature dim for the rest of the network

        # Encoder with ResNet50 backbone
        self.encoder = ResNetEncoder(pretrained_path, load_pretrained=load_encoder_pretrained)

        # FPN for multi-scale feature fusion (or channel-aligning fallback)
        if self.use_fpn:
            # ResNet50 feature channels: [256, 512, 1024, 2048] from stages 1-4
            self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=self.out_channels)
        else:
            # Fallback: take the deepest (stage4, 1/16 res) feature and project to out_channels
            self.fpn = nn.Sequential(
                nn.Conv2d(2048, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        # PPM for multi-scale spatial context aggregation (from PSPNet)
        if self.use_ppm:
            self.ppm = PyramidPoolingModule(in_channels=self.out_channels,
                                             pool_scales=(1, 2, 3, 6), reduction=4)
        else:
            self.ppm = nn.Identity()

        # Lightweight spatial-channel attention for support/query feature refinement
        self.sc_attn = LightweightSpatialChannelAttention(self.out_channels, reduction=16)
        if self.use_cross_attn:
            self.sq_bi_attn = LightweightBidirectionalSupportQueryAttention(
                self.out_channels, reduction=16, temperature=1.0, feedback_ratio=0.5
            )
        else:
            self.sq_bi_attn = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def _fuse_encoder_features(self, feat_multi):
        """
        Run FPN over ResNet's multi-scale outputs, or fall back to a single-feature path.

        Args:
            feat_multi: list of ResNet stage features [C1, C2, C3, C4]
        Returns:
            fused features with channel == self.out_channels
        """
        if self.use_fpn:
            return self.fpn(feat_multi)
        # Fallback: use only the deepest stage (stage4), same spatial size as ResNet output
        deepest = feat_multi[-1]  # [B, 2048, H/16, W/16]
        # use_fpn=False: self.fpn is already replaced with nn.Sequential fallback
        return self.fpn(deepest)

    def ablation_tag(self):
        """Return a short string tag describing the active ablation config."""
        return "FPN{}_PPM{}_XAttn{}".format(
            int(self.use_fpn), int(self.use_ppm), int(self.use_cross_attn)
        )

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images (way x shot x [B x 3 x H x W])
            fore_mask: foreground masks
            back_mask: background masks
            qry_imgs: query images (N x [B x 3 x H x W])
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        # Extract features with FPN and PPM
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        feat_multi = self.encoder(imgs_concat)
        fused = self._fuse_encoder_features(feat_multi)
        # Apply PPM for multi-scale spatial context aggregation
        ppm_enhanced = self.ppm(fused)
        # Apply lightweight spatial-channel attention
        img_fts = self.sc_attn(self.output_proj(ppm_enhanced))

        fts_size = img_fts.shape[-2:]

        # Split back into support and query features
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)

        # Process masks
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)

        # Compute predictions
        align_loss = 0
        outputs = []

        for epi in range(batch_size):
            supp_episode = supp_fts[:, :, epi]
            qry_episode = qry_fts[:, epi]

            supp_fg_fts = [[self.getFeatures(supp_episode[way, shot].unsqueeze(0), fore_mask[way, shot, [epi]])
                           for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_episode[way, shot].unsqueeze(0), back_mask[way, shot, [epi]])
                           for shot in range(n_shots)] for way in range(n_ways)]

            # Get prototypes
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            if self.sq_bi_attn is not None:
                # Build bidirectional support-query interaction from foreground prototypes
                fg_stack = torch.stack(fg_prototypes, dim=0)
                support_context = fg_stack.mean(dim=0).unsqueeze(-1).unsqueeze(-1)
                support_context = support_context.expand(-1, -1, qry_episode.shape[-2], qry_episode.shape[-1])
                support_context, qry_episode = self.sq_bi_attn(support_context, qry_episode)

            # Compute distance
            prototypes = [bg_prototype] + fg_prototypes
            dist = [self.calDist(qry_episode, prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            # Alignment loss
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_episode, pred, supp_episode,
                                               fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / batch_size

    def calDist(self, fts, prototype, scaler=20):
        """Calculate cosine similarity distance"""
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist

    def getFeatures(self, fts, mask):
        """Extract masked features"""
        # Ensure mask is 4D and single channel
        if mask.dim() == 4 and mask.shape[1] != 1:
            mask = mask[:, :1, :, :]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() == 4 and mask.shape[-1] in [3, 4]:
            mask = mask.permute(0, 3, 1, 2)[:, :1, :, :]

        # Resize mask to feature size
        mask = F.interpolate(mask.float(), size=fts.shape[-2:], mode='nearest')

        # Masked average pooling
        masked_fts = (fts * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)
        return masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        """Average features to get prototypes"""
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """Prototype alignment loss"""
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True)
        pred_mask = pred_mask.contiguous().int()

        binary_masks = []
        for i in range(1 + n_ways):
            mask_i = torch.eq(pred_mask, i).float()
            binary_masks.append(mask_i)

        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)

        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue

            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]

            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                fg = fore_mask[way, shot]
                bg = back_mask[way, shot]

                if fg.dim() == 4:
                    fg = fg.squeeze(0)
                if bg.dim() == 4:
                    bg = bg.squeeze(0)

                supp_label = torch.full_like(fg, 255, device=img_fts.device).long()
                supp_label[fg == 1] = 1
                supp_label[bg == 1] = 0
                supp_label = supp_label.unsqueeze(0)

                loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / n_shots / n_ways

        return loss
