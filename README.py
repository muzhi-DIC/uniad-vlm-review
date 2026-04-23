"""
UniAD VLM-AD 扩展：为 UniAD 端到端自动驾驶模型添加 VLM（视觉语言模型）三级语义对齐监督。

包含三个核心模块：
  losses/vlm_loss.py           — VLMAlignmentLoss & VLMContrastiveLoss
  detectors/uniad_e2e_vlm.py   — BEV/Planning/Motion 三级 VLM 损失集成
  dense_heads/planning_head.py — plan_query 输出键暴露
  tests/test_vlm_loss.py       — VLM 损失函数单元测试
"""
