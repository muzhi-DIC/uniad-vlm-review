"""
detectors/uniad_e2e_vlm.py
VLM 三级对齐损失计算模块（从 uniad_e2e.py 提取，供代码审查使用）

包含：
  - UniAD.__init__ 中的 VLM 监督模块初始化
  - UniAD.forward_train 中的 VLM 损失计算块（BEV / Planning / Motion 三级）

注意：本文件为代码审查目的的摘录，非可独立运行的完整模块。
"""
import traceback
import torch
import torch.nn as nn
from ..losses.vlm_loss import VLMAlignmentLoss


# ═══════════════════════════════════════════════════════════════════════════════
# __init__ 中的 VLM 监督模块初始化
# ═══════════════════════════════════════════════════════════════════════════════

def init_vlm_supervision(self, vlm_supervision):
    """在 UniAD.__init__ 中调用，初始化 VLM 三级对齐所需的投影层和损失函数。"""
    feat_path = vlm_supervision['feature_path']
    vlm_dim   = vlm_supervision.get('vlm_dim', 1536)
    vis_dim   = vlm_supervision.get('vis_dim', 256)
    vlm_loss_weight      = vlm_supervision.get('loss_weight', 0.5)
    planning_vlm_weight  = vlm_supervision.get('planning_loss_weight', 1.0)
    motion_vlm_weight    = vlm_supervision.get('motion_loss_weight', 0.3)

    # BEV 级别投影头：视觉规划特征 (vis_dim) -> VLM 特征空间 (vlm_dim)
    self.vlm_proj = nn.Sequential(
        nn.Linear(vis_dim, vis_dim * 2),
        nn.GELU(),
        nn.Linear(vis_dim * 2, vlm_dim),
    )

    # Planning 级别 VLM 投影（vis_dim -> vlm_dim）
    self.planning_vlm_proj = nn.Sequential(
        nn.Linear(vis_dim, vlm_dim),
        nn.LayerNorm(vlm_dim),
    )

    # Motion 级别 VLM 投影（vis_dim -> vlm_dim）
    self.motion_vlm_proj = nn.Sequential(
        nn.Linear(vis_dim, vlm_dim),
        nn.LayerNorm(vlm_dim),
    )

    # 三级损失函数（各自独立权重）
    self.vlm_loss_fn          = VLMAlignmentLoss(loss_weight=vlm_loss_weight)
    self.planning_vlm_loss_fn = VLMAlignmentLoss(loss_weight=planning_vlm_weight)
    self.motion_vlm_loss_fn   = VLMAlignmentLoss(loss_weight=motion_vlm_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# forward_train 中的 VLM 损失计算块
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vlm_losses(self, losses, bev_embed, outs_planning, outs_motion, img_metas):
    """
    在 UniAD.forward_train 中调用，计算 BEV / Planning / Motion 三级 VLM 对齐损失。

    Args:
        losses (dict): 当前损失字典，本函数会向其中添加 vlm.contrastive / vlm.planning / vlm.motion 键。
        bev_embed: BEVFormer 输出的 BEV 特征，形状通常为 [H*W, B, C]。
        outs_planning (dict): planning head 输出，包含 plan_query 等键。
        outs_motion (dict): motion head 输出，包含 sdc_traj_query 等键。
        img_metas (list[dict]): 图像元信息，包含 sample_idx 等字段。
    """
    if not (hasattr(self, 'with_vlm') and self.with_vlm and self.training):
        return

    # 从特征库中查找当前 batch 的 VLM 特征
    sample_tokens = [m['sample_idx'] for m in img_metas]
    vlm_feats, valid_mask = self.vlm_bank.lookup(sample_tokens, device=bev_embed.device)

    _vlm_feats = vlm_feats if valid_mask.sum() > 0 else None
    _vlm_mask  = valid_mask if valid_mask.sum() > 0 else None

    # ── 1. BEV 级别 VLM 对齐（BEV 特征全局平均 → 投影 → 对比）────────────
    try:
        if _vlm_feats is not None and _vlm_mask is not None:
            losses['vlm.contrastive'] = bev_embed.sum() * 0.0  # 默认 0
            if valid_mask.sum() > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    bev_f32 = bev_embed.float()
                    if bev_f32.dim() == 4:
                        vis_feat = bev_f32.flatten(2).mean(dim=-1)
                    elif bev_f32.dim() == 3:
                        # 启发式判断 BEV 形状：[H*W,B,C] vs [B,H*W,C]
                        if bev_f32.shape[0] > bev_f32.shape[1]:
                            vis_feat = bev_f32.mean(dim=0)
                        else:
                            vis_feat = bev_f32.mean(dim=1)
                    else:
                        vis_feat = bev_f32

                    vis_proj = self.vlm_proj.float()(vis_feat)
                    losses['vlm.contrastive'] = self.vlm_loss_fn(
                        vis_proj, vlm_feats.float(), valid_mask
                    )
    except Exception as e:
        # FAIL-2: except Exception 范围过宽，包含 CUDA OOM 等严重错误
        # 建议区分 (KeyError, AttributeError) 和其他异常
        print(f"[VLM Loss ERROR] {type(e).__name__}: {e}")

    # ── 2. Planning 级别 VLM 对齐（plan_query → 投影 → 对齐）────────────────
    if _vlm_feats is not None and _vlm_mask is not None and outs_planning is not None:
        try:
            plan_q = outs_planning['outs_motion'].get('plan_query', None)
            if plan_q is not None and _vlm_mask.sum() > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    plan_proj = self.planning_vlm_proj.float()(plan_q.float())
                    losses['vlm.planning'] = self.planning_vlm_loss_fn(
                        plan_proj, _vlm_feats.float(), _vlm_mask
                    )
        except Exception as e:
            # FAIL-2: 同上，except Exception 过宽
            print(f"[VLM Planning ERROR] {type(e).__name__}: {e}")

    # ── 3. Motion 级别 VLM 对齐（sdc_traj_query → 投影 → 对齐）─────────────
    if _vlm_feats is not None and _vlm_mask is not None and outs_motion:
        try:
            sdc_traj_q = outs_motion.get('sdc_traj_query', None)
            if sdc_traj_q is not None and _vlm_mask.sum() > 0:
                # FAIL-1: 缺少 len(sdc_traj_q) > 0 的边界检查
                # sdc_traj_q[-1] 在 sdc_traj_q 为空列表/空 tensor 时会抛 IndexError
                motion_feat = sdc_traj_q[-1].mean(dim=1)   # [B, D]
                with torch.cuda.amp.autocast(enabled=False):
                    motion_proj = self.motion_vlm_proj.float()(motion_feat.float())
                    losses['vlm.motion'] = self.motion_vlm_loss_fn(
                        motion_proj, _vlm_feats.float(), _vlm_mask
                    )
        except Exception as e:
            # FAIL-2: except Exception 过宽
            print(f"[VLM Motion ERROR] {type(e).__name__}: {e}")
