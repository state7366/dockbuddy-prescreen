# dockbuddy-prescreen

DockBuddy 突变预筛管线，从 `pipeline/prescreen` 抽出的解耦版本。

## 职责

业务层编排：打分 → 布尔闸 → 多样性采样 → 折叠回测 → 精筛 → 导出（EVOLVEpro）。

依赖（无环）：`dockbuddy-common` + `dockbuddy-scoring` + `dockbuddy-mutate` + `dockbuddy-evolution`。

## 快速开始

```python
from dockbuddy_prescreen.screen import screen_mutations
candidates = screen_mutations(sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP")
```

## 组件

| 模块 | 职责 |
|---|---|
| `mutlib.py` | 突变库生成（Variant / MutationSet schema） |
| `gate.py` | 布尔硬约束闸 |
| `diverse.py` | 多样性采样 |
| `store.py` | 候选状态机 |
| `export.py` | EVOLVEpro 导出 |
| `feedback.py` | 失败日志 + 湿实验反馈 |
| `fold_api.py` | 折叠回测（ESMFold 双通道） |
| `orchestrator.py` | 编排（run_pipeline） |
| `screen.py` | 一键 API（screen_mutations） |
