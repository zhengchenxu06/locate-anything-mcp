# LocateAnything MCP Server

👁️ **给 Claude Code 装上"眼睛"** — 基于 NVIDIA LocateAnything-3B 的视觉定位 MCP Server。

截图 + 自然语言描述 → 精确像素坐标。解决 CSS 选择器对动态 UI / Canvas / 不可见元素失效的问题。

## 架构

```
Claude Code (DeepSeek v4 Pro)
    │
    ├── Playwright MCP (截图/点击)
    └── LocateAnything MCP (视觉定位) ← 本项目
         │ HTTP localhost:8765
         ▼
    WSL2 Ubuntu — LocateAnything-3B (INT4, ~1s/次)
```

## 快速开始

### 前置条件

- Windows 11 + WSL2 (Ubuntu 22.04)
- NVIDIA GPU (8GB+ VRAM)，本项目使用 RTX 4060 Laptop
- 模型权重下载到 WSL2 `/home/locate-anything-model/`（从 [Gitee 镜像](https://gitee.com/hf-models/LocateAnything-3B) 下载）

### 安装

```bash
# 1. Windows 侧安装依赖
pip install fastmcp requests pyyaml Pillow

# 2. WSL2 侧安装依赖
wsl -d Ubuntu -- pip3 install fastapi uvicorn transformers torch bitsandbytes accelerate Pillow

# 3. 注册 MCP Server
claude mcp add locate-anything -- python "C:\Users\Administrator\locate-anything-mcp\mcp_server.py"
```

### 启动

```bash
# 1. 先启动 WSL2 Worker
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/Administrator/locate-anything-mcp && setsid python3 wsl_worker.py &>/tmp/worker.log & disown"

# 2. 验证
curl http://localhost:8765/health
# → {"status":"ok","model_loaded":true,"gpu_available":true}

# 3. 重启 Claude Code 使 MCP 工具生效
```

## 工具

### ground_gui — 单目标定位

```
输入: image_b64 (截图base64), description (自然语言描述), mode ("fast"|"hybrid")
输出: {boxes: [{bbox: [x1,y1,x2,y2], score: 1.0}]}
```

坐标是 0-1000 量化值，需乘以图像实际宽高换算像素。

### locate_all — 批量检测

```
输入: image_b64, categories (如 ["按钮", "输入框"]), mode
输出: {boxes_by_category: {"按钮": [...], "输入框": [...]}}
```

## 技术栈

- **模型**: [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) (CVPR 2026)
- **架构**: MoonViT-SO-400M + Qwen2.5-3B-Instruct + PBD 并行解码
- **推理**: INT4 量化 (bitsandbytes NF4), Fast 模式 ~1s/次
- **框架**: FastMCP + FastAPI + Transformers

## 许可

本项目代码使用 MIT License。模型权重使用 NVIDIA 非商业许可。
