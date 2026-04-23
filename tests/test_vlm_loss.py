"""
实验一：服务单元测试
被测模块：projects/mmdet3d_plugin/losses/vlm_loss.py
覆盖类：VLMAlignmentLoss、VLMContrastiveLoss

测试目标：
  - 正常路径：对齐损失数值正确
  - 边界条件：valid_mask 全无效、N=1、loss_weight 缩放
  - 数值特性：输出范围 [0, 2]、reduction 模式
  - VLMContrastiveLoss：InfoNCE 对比逻辑及单样本降级
"""

import pytest
import torch
import torch.nn.functional as F

# ----- 被测模块导入 -----
from mmdet3d_plugin.losses.vlm_loss import VLMAlignmentLoss, VLMContrastiveLoss


# ============================================================
# 辅助工具
# ============================================================

def _randn(*shape):
    """固定随机种子生成，保证测试可复现"""
    torch.manual_seed(42)
    return torch.randn(*shape)


# ============================================================
# VLMAlignmentLoss 测试
# ============================================================

class TestVLMAlignmentLoss:
    """VLMAlignmentLoss 单元测试"""

    def test_perfect_alignment_loss_near_zero(self):
        """完全对齐时 loss ≈ 0：vision_feat == vlm_feat"""
        loss_fn = VLMAlignmentLoss(loss_weight=1.0)
        feat = _randn(4, 256)
        loss = loss_fn(feat.clone(), feat.clone())
        assert loss.item() < 1e-5, f"Expected ~0, got {loss.item():.6f}"

    def test_opposite_directions_loss_near_two(self):
        """完全反向时 cosine=-1 → loss ≈ 2"""
        loss_fn = VLMAlignmentLoss(loss_weight=1.0)
        feat = _randn(4, 256)
        loss = loss_fn(feat.clone(), -feat.clone())
        assert abs(loss.item() - 2.0) < 1e-4, f"Expected ~2.0, got {loss.item():.6f}"

    def test_loss_range_zero_to_two(self):
        """loss 值必须落在 [0, 2] 范围内"""
        loss_fn = VLMAlignmentLoss()
        v = _randn(8, 512)
        t = _randn(8, 512)
        loss = loss_fn(v, t)
        assert 0.0 <= loss.item() <= 2.0, f"Loss out of range [0,2]: {loss.item():.4f}"

    def test_valid_mask_filters_samples(self):
        """valid_mask 过滤后与直接传入有效样本结果一致"""
        loss_fn = VLMAlignmentLoss()
        torch.manual_seed(7)
        v = torch.randn(4, 256)
        t = torch.randn(4, 256)
        mask = torch.tensor([True, False, True, False])

        loss_masked = loss_fn(v.clone(), t.clone(), mask)
        loss_direct = loss_fn(v[mask].clone(), t[mask].clone())

        assert torch.allclose(loss_masked, loss_direct, atol=1e-5), (
            f"Masked loss {loss_masked.item():.6f} != direct loss {loss_direct.item():.6f}"
        )

    def test_all_invalid_mask_returns_zero_tensor(self):
        """valid_mask 全为 False 时，应返回 0（梯度可传播的零标量）"""
        loss_fn = VLMAlignmentLoss()
        v = _randn(4, 256).requires_grad_(True)
        t = _randn(4, 256)
        mask = torch.zeros(4, dtype=torch.bool)
        loss = loss_fn(v, t, mask)
        assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"

    def test_loss_weight_linear_scaling(self):
        """loss_weight=w 时，损失值 = w * loss_weight_1 的结果"""
        torch.manual_seed(99)
        v = torch.randn(4, 256)
        t = torch.randn(4, 256)
        l1 = VLMAlignmentLoss(loss_weight=1.0)(v.clone(), t.clone())
        l2 = VLMAlignmentLoss(loss_weight=2.0)(v.clone(), t.clone())
        l5 = VLMAlignmentLoss(loss_weight=5.0)(v.clone(), t.clone())
        assert torch.allclose(l2, 2.0 * l1, atol=1e-5), "2x weight scaling failed"
        assert torch.allclose(l5, 5.0 * l1, atol=1e-5), "5x weight scaling failed"

    def test_single_sample_n1_works(self):
        """N=1 单样本情况下不报错，loss 在合法范围内"""
        loss_fn = VLMAlignmentLoss()
        v = _randn(1, 256)
        t = _randn(1, 256)
        loss = loss_fn(v, t)
        assert 0.0 <= loss.item() <= 2.0

    def test_reduction_sum_vs_mean(self):
        """reduction='sum' 时 loss = mean * N"""
        torch.manual_seed(5)
        v = torch.randn(4, 256)
        t = torch.randn(4, 256)
        loss_mean = VLMAlignmentLoss(reduction='mean')(v.clone(), t.clone())
        loss_sum  = VLMAlignmentLoss(reduction='sum')(v.clone(), t.clone())
        assert torch.allclose(loss_sum, loss_mean * 4, atol=1e-5), (
            f"sum={loss_sum.item():.6f}, mean*4={loss_mean.item()*4:.6f}"
        )

    def test_gradient_flows_through_loss(self):
        """loss 应可反向传播（梯度不为 None）"""
        loss_fn = VLMAlignmentLoss()
        v = _randn(2, 256).requires_grad_(True)
        t = _randn(2, 256)
        loss = loss_fn(v, t)
        loss.backward()
        assert v.grad is not None, "No gradient computed for vision_feat"
        assert not torch.all(v.grad == 0), "Gradient is all zeros"


# ============================================================
# VLMContrastiveLoss 测试
# ============================================================

class TestVLMContrastiveLoss:
    """VLMContrastiveLoss (InfoNCE) 单元测试"""

    def test_infonce_with_batch_n4(self):
        """N≥2 时 InfoNCE 正常计算，loss > 0"""
        loss_fn = VLMContrastiveLoss(temperature=0.07)
        v = _randn(4, 256)
        t = _randn(4, 256)
        loss = loss_fn(v, t)
        assert loss.item() > 0.0

    def test_infonce_falls_back_for_n1(self):
        """N=1 时退化为 alignment loss，结果与 VLMAlignmentLoss 一致"""
        torch.manual_seed(11)
        v = torch.randn(1, 256)
        t = torch.randn(1, 256)
        loss_contrast = VLMContrastiveLoss(loss_weight=1.0)(v.clone(), t.clone())
        loss_align    = VLMAlignmentLoss(loss_weight=1.0)(v.clone(), t.clone())
        assert torch.allclose(loss_contrast, loss_align, atol=1e-5), (
            f"N=1 fallback mismatch: contrast={loss_contrast.item():.6f}, align={loss_align.item():.6f}"
        )

    def test_perfect_alignment_low_infonce(self):
        """当 v == t 时（完全匹配），InfoNCE 损失应为对角线最优，接近 log(N)"""
        N = 4
        feat = _randn(N, 256)
        loss_fn = VLMContrastiveLoss(temperature=0.07, loss_weight=1.0)
        loss = loss_fn(feat.clone(), feat.clone())
        # 完美对齐时对角线 logit 极大 → 损失接近 0
        assert loss.item() < 0.5, f"Ideal alignment loss too large: {loss.item():.4f}"

    def test_contrastive_weight_scaling(self):
        """loss_weight 对 InfoNCE 同样生效"""
        torch.manual_seed(3)
        v = torch.randn(4, 256)
        t = torch.randn(4, 256)
        l1 = VLMContrastiveLoss(loss_weight=1.0)(v.clone(), t.clone())
        l3 = VLMContrastiveLoss(loss_weight=3.0)(v.clone(), t.clone())
        assert torch.allclose(l3, 3.0 * l1, atol=1e-5)
