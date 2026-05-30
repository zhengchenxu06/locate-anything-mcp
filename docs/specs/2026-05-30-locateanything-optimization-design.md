# LocateAnything MCP Server 优化设计

> 设计文档 | 2026-05-30 | 基于 NVIDIA 官方源码最佳实践

## 概述

对照 NVIDIA 官方 `generate_utils.py` 的算法细节，优化 `wsl_worker.py` 的三项短板：置信度输出、坐标加权平均与空框检测、智能模式切换。

---

## A. 真实置信度输出

### 现状
`_parse_boxes()` 写死 `score: 1.0`，丢弃了模型内部的概率信息。

### 改动
`predict()` 调用 `model.generate(verbose=True)`，返回的 tuple 第 3 位 `stats` 包含每步采样信息的 `sampling_history`。从中提取每个 block 的 token 置信度。

输出结构新增 `score` 字段，取 4 个坐标位置置信度的均值：
```json
{"bbox": [396, 370, 608, 502], "score": 0.93}
```

---

## B. 坐标加权平均 + empty_box

### 改动 1: empty_box 识别
`_parse_boxes()` 检测 `<box>none</box>` 或相关模式，识别模型明确表示"无目标"的情况。
输出结构新增 `empty_detected: bool`。

### 改动 2: 坐标加权平均
NVIDIA `decode_bbox_avg()` 对每个坐标位置取 top-4 候选做加权平均，比 argmax 更平滑。在解析 `raw_answer` 时保留这一结果。同时在 `predict()` 输出中附带 `confidence` 信息。

---

## C. "auto" 模式两级兜底

### 新增 mode: "auto"
作为默认模式，策略：
1. 先用 Fast 模式推理（~1s）
2. 若 `score < 0.5` 或 `empty_detected=true` → 自动用 Hybrid 重试
3. 返回最终结果 + 标记

输出结构：
```json
{
  "boxes": [...],
  "score": 0.83,
  "empty_detected": false,
  "mode_used": "hybrid",
  "retried": true,
  "fast_result": {"score": 0.31, "boxes": []}
}
```

### mode 参数更新
| mode | 行为 |
|------|------|
| `"fast"` | 纯 MTP，最快 |
| `"hybrid"` | MTP + AR 兜底，模型内部切换 |
| `"auto"`（新增，默认） | Fast 先跑，低置信度时自动用 Hybrid 重试 |

---

## 改动范围

| 文件 | 改动 |
|------|------|
| `wsl_worker.py` | `predict()` → 启用 verbose 模式提取 stats |
| | `_parse_boxes()` → empty_box 识别 |
| | `ground_gui()` / `locate_all()` → auto 模式兜底 |
| | 返回结构升级（score, empty_detected, mode_used, retried） |

不改动：
- `mcp_server.py` — 透传
- `bridge.py` — 透传
- 模型权重 — 不涉及
- 配置文件 — 不涉及

---

## 验证方式

```bash
# 1. 合成图测试 empty_box
# 纯色图 + "find a person" → empty_detected=true, boxes=[]

# 2. 合成图测试置信度
# 红色按钮 + "the red button" → score > 0.8

# 3. auto 模式压力测试
# 模糊小目标 → Fast score < 0.5 → 自动切 Hybrid → 返回 retried=true
```
