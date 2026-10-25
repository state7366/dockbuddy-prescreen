# -*- coding: utf-8 -*-
"""
dockbuddy_prescreen —— 突变预筛管线(业务层 · 编排闭环的组件库)
================================================================================
把预筛管线相关的库从 pipeline 顶层平摊收敛到本独立包。主仓库 pipeline/prescreen 保留薄 shim。

职责边界（业务层 orchestration 只做调度/状态/落盘，打分/几何口径各居其位）：
  mutlib.py     突变库生成（饱和突变 / 位置规格 / 变体列表 / MutationSet schema）
  gate.py       布尔硬约束闸（阈值全外置，扁平阈值表）
  diverse.py    多样性采样（site + aa_type 贪心最远点）
  store.py      候选状态机（pending→scored→passed_gate→selected→exported→wet_feedback）
  export.py     EVOLVEpro 导出（label_source=proxy，不预测 kcat/Km）
  feedback.py   失败日志 + 湿实验反馈对撞
  fold_api.py   折叠回测 API（ESMFold 双通道：api 公共 API / local GPU 本地部署）
  orchestrator.py  业务层编排（run_pipeline：打分→闸→采样→折叠回测→精筛→导出）
  screen.py     应用层一键 API（screen_mutations：序列→少量湿实验候选）

新式访问（推荐）：`from dockbuddy_prescreen.gate import evaluate`。
"""
__version__ = "0.1.0"
