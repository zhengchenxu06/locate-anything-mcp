# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LocateAnything-3B 视觉定位 MCP Server — 为 Claude 提供"眼睛"能力。截图 + 自然语言描述 → 精确像素坐标，解决 CSS 选择器对动态 UI / Canvas / 不可见元素失效的问题。

## 架构

```
Windows (MCP Server)              WSL2 Ubuntu (推理 Worker)
┌──────────────────────┐          ┌──────────────────────────┐
│ mcp_server.py        │  HTTP    │ wsl_worker.py            │
│   FastMCP 工具:       │ ──────→ │   FastAPI :8765          │
│   - ground_gui       │ localhost│   LocateAnything-3B INT4 │
│   - locate_all       │ ←────── │   (3.1GB / 8.6GB VRAM)   │
│                      │   JSON   │   Fast 模式: ~1s/次       │
└──────────────────────┘          └──────────────────────────┘
         │
    bridge.py (HTTP 通信层)
```

- `mcp_server.py` — FastMCP 入口，向 Claude 暴露 2 个工具
- `bridge.py` — HTTP 通信桥，转发请求到 WSL2 Worker
- `wsl_worker.py` — WSL2 侧 FastAPI 服务，常驻模型 + 推理
- `config.yaml` — 推理参数配置

## 启动与停止

### 启动 WSL2 Worker（必须先启动）

```bash
# 从 Windows 终端（管理员）:
wsl.exe -d Ubuntu -- bash -c "cd /mnt/c/Users/Administrator/locate-anything-mcp && setsid python3 wsl_worker.py &>/tmp/worker.log & disown"
```

### 验证 Worker

```bash
curl http://localhost:8765/health
# → {"status":"ok","model_loaded":true,"gpu_available":true,"vram_gb":8.6}
```

### MCP Server

由 Claude Code 自动启动（通过 `claude mcp add` 注册的 stdio 服务器），无需手动干预。

## 两个 MCP 工具

### ground_gui — 单目标定位

```
输入: image_b64 (截图base64), description (自然语言描述), mode ("auto"|"fast"|"hybrid")
输出:
{
  "raw_answer": "...",
  "boxes": [{"bbox": [396, 370, 608, 502], "score": 0.9}],
  "empty_detected": false,
  "mode_used": "auto",
  "retried": false
}
```

坐标是 0-1000 量化值，需乘以图像实际宽高换算像素：

**mode 参数：**

| mode | 行为 |
|------|------|
| `"auto"` (默认) | Fast 先跑，低置信度自动切 Hybrid |
| `"fast"` | 纯 MTP，最快 |
| `"hybrid"` | MTP + AR 兜底，最高精度 |

**响应字段：**
- `score`: 置信度分数 (0-1)，由模型 logit 计算
- `empty_detected`: 是否检测到空框/无目标
- `mode_used`: 实际使用的推理模式
- `retried`: Auto 模式下是否触发了 Hybrid 重试
- `fast_result`: Auto 模式下 Fast 阶段的原始结果（仅 retried=true 时存在）
```python
px = coord * image_width // 1000
```

典型用法：
```
1. Playwright 截图 → base64
2. ground_gui(screenshot, "蓝色提交按钮")
3. 拿坐标 → elementFromPoint(x, y).click()
```

### locate_all — 批量检测

```
输入: image_b64, categories (如 ["按钮", "输入框"]), mode
输出: {boxes_by_category: {"按钮": [...], "输入框": [...]}}
```

## 关键设计决策

- **INT4 量化（必须）**: RTX 4060 8GB 装不下 BF16 模型（7.14GB），bitsandbytes NF4 量化后仅 3.1GB。CPU 推理会 OOM（WSL2 内存限制）。
- **坐标格式**: 模型输出 `<box><x1><y1><x2><y2></box>` 格式（每个坐标是独立 token，不是逗号分隔），正则 `<box><(\d+)><(\d+)><(\d+)><(\d+)></box>` 解析。
- **Fast 模式**: PBD 并行解码，6个 token 同时预测，比 AR 快 3-6 倍。适合 Agent 实时场景。
- **setsid 启动**: WSL2 进程随 bash 退出被杀，`setsid ... & disown` 可彻底脱离终端。

## 常见问题

### Worker 连不上 (Connection Refused)
WSL2 重启后 Worker 不会自动启动，需手动执行启动命令。

### 推理超时 (>30s)
模型可能在 CPU 上运行。检查 `curl http://localhost:8765/health` 确认 `gpu_available: true`。

### 坐标不准
- description 描述不够精确 → 加更多细节（颜色、位置、文字）
- Fast 模式精度略低 → 切 `mode: "hybrid"` 重试
- 图像分辨率超过 2.5K → bridge.resize_if_needed() 会自动缩放

### MCP 工具不出现
```bash
# 检查注册状态
powershell -NoProfile -Command "claude mcp list"
# 重新注册（用 PowerShell 避免反斜杠被吃）
powershell -NoProfile -Command "claude mcp add locate-anything -- python 'C:\Users\Administrator\locate-anything-mcp\mcp_server.py'"
```
然后完全重启 Claude Code。

## 模型信息

- **模型**: nvidia/LocateAnything-3B (CVPR 2026)
- **架构**: MoonViT-SO-400M + Qwen2.5-3B-Instruct + PBD 并行解码
- **WSL2 路径**: `/home/locate-anything-model/`
- **权重**: model-00001 (4.7G) + model-00002 (2.6G) safetensors
- **来源**: Gitee 镜像 `https://gitee.com/hf-models/LocateAnything-3B`
- **LFS 下载**: Gitee 不支持 git-lfs pull，需用 `?lfs=1` 参数直接下载原始文件
