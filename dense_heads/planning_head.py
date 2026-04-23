"""
dense_heads/planning_head.py (摘录)
plan_query 输出键暴露 — VLM Planning 级对齐所需
"""
import torch
import torch.nn as nn


class PlanningHeadSingleMode(nn.Module):
    """
    规划头（单模式），返回 dict 包含 plan_query 键供 VLM Planning 对齐使用。

    forward() 返回值中 'plan_query' 的形状约定：
      - batch_size = 1（训练时单样本）：shape = (1, embed_dims)
      - batch_size = B（> 1）：shape = (B, embed_dims)
    注意：squeeze(0) 在 B > 1 时会移除批次维度的第0维，请调用方验证。
    """

    def forward(self, bev_embed, occ_mask, bev_pos,
                sdc_traj_query, sdc_track_query, command):
        """
        Args:
            bev_embed       : [H*W, B, D]
            sdc_traj_query  : [n_dec, B, n_mode, D]
            sdc_track_query : [B, D]
            command         : list[int]，长度为 B

        Returns:
            dict with keys: 'sdc_traj', 'sdc_traj_all', 'plan_query'
        """
        # ... (omitted: BEV attention, trajectory decoding) ...

        # plan_query: 提取规划决策的语义嵌入，供 VLM Planning 对齐
        # 形状：squeeze(0) 后为 [B, embed_dims]（当 B=1 时为 [1, embed_dims]）
        plan_query = sdc_traj_query[-1].mean(dim=-2).squeeze(0)  # [B, D]

        return {
            'sdc_traj': sdc_traj_query[-1].mean(dim=-2),
            'sdc_traj_all': sdc_traj_query,
            'plan_query': plan_query,   # 供 VLM Planning 级对齐使用 [B, D]
        }
