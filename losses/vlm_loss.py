"""
VLM 监督损失模块

提供两种损失：
1. VLMContrastiveLoss  - InfoNCE（需要 batch_size ≥ 2 个有效样本）
2. VLMAlignmentLoss    - Cosine Alignment（单样本即可工作，适合训练场景）

默认使用 VLMAlignmentLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class VLMAlignmentLoss(nn.Module):
    """
    BEV 特征与 VLM 特征的余弦对齐损失
    
    loss = 1 - cosine_similarity(vision_proj, vlm_feat)
    
    对单样本 (N=1) 同样有效，适合 batch_size=1 的蒸馏场景
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean'):
        super(VLMAlignmentLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, vision_feat, vlm_feat, valid_mask=None):
        """
        Args:
            vision_feat: [B, D] 视觉特征（投影后）
            vlm_feat:    [B, D] VLM 语义特征
            valid_mask:  [B] bool，是否有有效 VLM 特征

        Returns:
            loss: scalar，范围 [0, 2]
        """
        if valid_mask is not None:
            valid = valid_mask.bool()
            if valid.sum() == 0:
                return vision_feat.sum() * 0.0
            vision_feat = vision_feat[valid]
            vlm_feat = vlm_feat[valid]

        # L2 归一化
        v = F.normalize(vision_feat, dim=-1)   # [N, D]
        t = F.normalize(vlm_feat, dim=-1)       # [N, D]

        # 余弦相似度 [-1, 1] -> 对齐损失 [0, 2]
        cos_sim = (v * t).sum(dim=-1)           # [N]

        if self.reduction == 'mean':
            loss = (1.0 - cos_sim).mean()
        elif self.reduction == 'sum':
            loss = (1.0 - cos_sim).sum()
        else:
            loss = (1.0 - cos_sim).mean()

        return loss * self.loss_weight


@LOSSES.register_module()
class VLMContrastiveLoss(nn.Module):
    """
    跨模态对比损失 (InfoNCE)
    
    注意：需要 batch 内至少 2 个有效样本，否则损失退化为 0
    适合 batch_size 较大的场景
    """

    def __init__(self,
                 temperature=0.07,
                 loss_weight=1.0,
                 reduction='mean'):
        super(VLMContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, vision_feat, vlm_feat, valid_mask=None):
        """
        Args:
            vision_feat: [B, D] 视觉规划特征（投影后）
            vlm_feat:    [B, D] VLM 语义特征（L2 归一化）
            valid_mask:  [B] bool，是否有有效 VLM 特征

        Returns:
            loss: scalar
        """
        if valid_mask is not None:
            valid = valid_mask.bool()
            if valid.sum() == 0:
                return vision_feat.sum() * 0.0
            vision_feat = vision_feat[valid]
            vlm_feat = vlm_feat[valid]

        N = vision_feat.shape[0]
        if N < 2:
            # 单样本时 InfoNCE 退化为 0，改用 alignment loss
            v = F.normalize(vision_feat, dim=-1)
            t = F.normalize(vlm_feat, dim=-1)
            return (1.0 - (v * t).sum(dim=-1)).mean() * self.loss_weight

        # L2 归一化
        v = F.normalize(vision_feat, dim=-1)
        t = F.normalize(vlm_feat, dim=-1)

        logits = torch.matmul(v, t.T) / self.temperature
        labels = torch.arange(N, device=logits.device)

        loss_v2t = F.cross_entropy(logits, labels, reduction=self.reduction)
        loss_t2v = F.cross_entropy(logits.T, labels, reduction=self.reduction)
        loss = (loss_v2t + loss_t2v) * 0.5

        return loss * self.loss_weight

