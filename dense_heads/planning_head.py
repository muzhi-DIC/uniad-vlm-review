"""dense_heads/planning_head.py (摘录) — plan_query 输出键暴露"""
import torch
import torch.nn as nn


class PlanningHeadSingleMode(nn.Module):
    """
    规划头（单模式），forward() 返回 plan_query 供 VLM Planning 对齐使用。

    plan_query 输出 shape 约定：
      - batch_size=1（训练）：shape = (1, embed_dims)
      - batch_size=B（>1）：shape = (B, embed_dims)
    注意：内部 squeeze(0) 仅在 B=1 时不影响 batch 维度，B>1 时需验证。
    """

    def forward(self, bev_embed, occ_mask, bev_pos,
                sdc_traj_query, sdc_track_query, command):
        # ... (BEV attention, trajectory decoding omitted) ...

        # plan_query: [B, embed_dims]，供 VLM Planning 级对齐
        plan_query = sdc_traj_query[-1].mean(dim=-2).squeeze(0)

        return {
            'sdc_traj':     sdc_traj_query[-1].mean(dim=-2),
            'sdc_traj_all': sdc_traj_query,
            'plan_query':   plan_query,
        }
