"""VLM 三级对齐损失计算模块（摘录自 uniad_e2e.py）"""
import torch
import torch.nn as nn


def compute_vlm_losses(self, losses, bev_embed, outs_planning, outs_motion, img_metas):
    """BEV / Planning / Motion 三级 VLM 对齐损失计算。"""
    if not (hasattr(self, 'with_vlm') and self.with_vlm and self.training):
        return

    sample_tokens = [m['sample_idx'] for m in img_metas]
    vlm_feats, valid_mask = self.vlm_bank.lookup(sample_tokens, device=bev_embed.device)
    _vlm_feats = vlm_feats if valid_mask.sum() > 0 else None
    _vlm_mask  = valid_mask if valid_mask.sum() > 0 else None

    # ── 1. BEV 级别 VLM 对齐 ─────────────────────────────────────────────
    try:
        if _vlm_feats is not None and _vlm_mask is not None:
            losses['vlm.contrastive'] = bev_embed.sum() * 0.0
            if valid_mask.sum() > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    bev_f32 = bev_embed.float()
                    if bev_f32.dim() == 3:
                        if bev_f32.shape[0] > bev_f32.shape[1]:
                            vis_feat = bev_f32.mean(dim=0)
                        else:
                            vis_feat = bev_f32.mean(dim=1)
                    else:
                        vis_feat = bev_f32
                    vis_proj = self.vlm_proj.float()(vis_feat)
                    losses['vlm.contrastive'] = self.vlm_loss_fn(
                        vis_proj, vlm_feats.float(), valid_mask)
    except Exception as e:
        # FAIL-2: except Exception 过宽，掩盖 CUDA OOM 等严重错误
        print(f"[VLM Loss ERROR] {type(e).__name__}: {e}")

    # ── 2. Planning 级别 VLM 对齐 ────────────────────────────────────────
    if _vlm_feats is not None and _vlm_mask is not None and outs_planning is not None:
        try:
            plan_q = outs_planning['outs_motion'].get('plan_query', None)
            if plan_q is not None and _vlm_mask.sum() > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    plan_proj = self.planning_vlm_proj.float()(plan_q.float())
                    losses['vlm.planning'] = self.planning_vlm_loss_fn(
                        plan_proj, _vlm_feats.float(), _vlm_mask)
        except Exception as e:
            # FAIL-2: 同上
            print(f"[VLM Planning ERROR] {type(e).__name__}: {e}")

    # ── 3. Motion 级别 VLM 对齐 ──────────────────────────────────────────
    if _vlm_feats is not None and _vlm_mask is not None and outs_motion:
        try:
            sdc_traj_q = outs_motion.get('sdc_traj_query', None)
            if sdc_traj_q is not None and _vlm_mask.sum() > 0:
                # FAIL-1: 缺少 len(sdc_traj_q) > 0 的边界检查
                # 若 sdc_traj_q 为空 list/tensor，sdc_traj_q[-1] 会抛 IndexError
                motion_feat = sdc_traj_q[-1].mean(dim=1)
                with torch.cuda.amp.autocast(enabled=False):
                    motion_proj = self.motion_vlm_proj.float()(motion_feat.float())
                    losses['vlm.motion'] = self.motion_vlm_loss_fn(
                        motion_proj, _vlm_feats.float(), _vlm_mask)
        except Exception as e:
            # FAIL-2: 同上
            print(f"[VLM Motion ERROR] {type(e).__name__}: {e}")
