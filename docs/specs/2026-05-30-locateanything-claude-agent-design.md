# LocateAnything-3B + Claude Agent 视觉定位集成方案

> 设计文档 | 2026-05-30 | 状态：已批准

## 概述

将 NVIDIA LocateAnything-3B 视觉定位模型集成为 Claude Code 的 MCP Server，解决 Claude + Playwright 在动态 UI、Canvas 渲染、无标签元素等场景下 CSS 选择器失效的问题。

**核心思路**：截图 + 自然语言描述 → 精确坐标 → Playwright 执行。

---

## 架构

```
Windows 宿主 (RTX 4060 Laptop GPU, 8GB VRAM)
┌─────────────────────────────────────────────────┐
│  Claude Code (DeepSeek v4 Pro)                   │
│  决策层：任务规划 + 目标描述 + 结果验证            │
│                                                  │
│  MCP Server (Windows 侧 Python 进程)              │
│  ├── 工具: ground_gui(screenshot, desc)          │
│  └── 工具: locate_all(screenshot, categories)    │
│       │                                           │
│       │ HTTP localhost:8765                       │
│       ▼                                           │
│  ┌──────────────────────────────────────────┐    │
│  │  WSL2 (Ubuntu 22.04)                     │    │
│  │  FastAPI Worker                           │    │
│  │  └── LocateAnything-3B Worker (常驻)      │    │
│  │      ├── moonvit 视觉编码器                │    │
│  │      ├── qwen2.5-3b 语言模型              │    │
│  │      └── PBD 并行解码 (Fast 模式)          │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  Playwright MCP (现有)                            │
│  └── 截图 / 点击 / 输入 / 导航                    │
└─────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 部署位置 | WSL2 (GPU-PV 直通) | NVIDIA 仅支持 Linux，WSL2 提供 GPU 直通 |
| 通信协议 | HTTP localhost | WSL2 ↔ Windows localhost 自动互通 |
| 推理模式 | Fast 默认 + Hybrid 兜底 | Fast ~150ms，异常时自动回退 |
| 模型加载 | 常驻 Worker | 避免每次推理重新加载 |
| 工具数量 | 2 个 (ground_gui + locate_all) | 覆盖 90% Web 自动化需求 |

---

## MCP 工具定义

### ground_gui

```
输入:
  image_b64: string      # 截图的 base64 编码
  description: string    # 自然语言描述，如 "蓝色提交按钮"
  mode: string           # "fast"(默认) | "hybrid"

输出:
  boxes: [{bbox: [x1,y1,x2,y2], score: 0.95}]
  raw_answer: string     # 原始模型输出 (调试用)
```

### locate_all

```
输入:
  image_b64: string
  categories: string[]   # 如 ["按钮", "输入框", "菜单项"]
  mode: string

输出:
  boxes_by_category: {category: [{bbox, score}]}
```

---

## Claude 交互时序

```
任务: "点击登录按钮"
│
├─ 1. 决策 (~500ms)        Claude 判断需要视觉定位
├─ 2. 截图 (~100ms)        Playwright screenshot
├─ 3. 定位 (~150ms)        ground_gui(screenshot, "login button")
├─ 4. 验证 (~200ms)        Claude 判断坐标合理性
├─ 5. 执行 (~50ms)         Playwright clickAt(x, y)
├─ 6. 确认                 截图比对，不符则重试
│
总计: ~1000ms (视觉定位仅占 15%)
```

### 工具选择决策

```
Claude 需要定位元素:
├─ 明确 CSS/文本 → Playwright 直接操作
├─ 动态 UI / Canvas / 无标签 → ground_gui
├─ 需要批量检测 → locate_all
└─ 定位失败 → 重试 (换描述 → 换模式 → 求助用户)
```

---

## 容错策略

| 场景 | 处理 |
|------|------|
| 未找到目标 | 返回 [] + score:0, Claude 决策重试 |
| WSL2 未启动 | MCP 启动检查 /health，不通则提示 |
| 模型 OOM | 自动 empty_cache + 降级 CPU |
| 图像过大 (>2.5K) | bridge 层自动缩放 |
| 推理超时 (>5s) | 返回超时错误，Claude 重试 |
| Fast 精度不足 | Claude 自动切 Hybrid 或换描述 |

---

## 文件结构

```
locate-anything-mcp/
├── mcp_server.py           # FastMCP 入口
├── bridge.py               # Windows ↔ WSL2 HTTP 通信
├── wsl_worker.py            # WSL2 侧 FastAPI Worker
├── locate_worker.py        # 官方 LocateAnythingWorker
├── generate_utils.py       # PBD 采样算法
├── modeling_locateanything.py  # 模型定义
├── configuration_locateanything.py
├── processing_locateanything.py
├── config.yaml             # 配置
└── requirements.txt
```

---

## 部署步骤

1. **导入 WSL2 Ubuntu** — `wsl --import Ubuntu-22.04 D:\WSL\Ubuntu-22.04 rootfs.tar.gz`
2. **WSL2 内安装环境** — CUDA, PyTorch, FastAPI, Transformers
3. **拉取模型** — 从 Gitee 镜像克隆 `nvidia/LocateAnything-3B`
4. **启动 Worker** — `python wsl_worker.py` 验证 /health
5. **安装 MCP** — `mcp add locate-anything`
6. **端到端测试** — Claude 截图 → 定位 → 点击

### 开发分阶段

| 阶段 | 内容 | 验证方式 |
|------|------|---------|
| 1. 最小可行 | WSL2 Worker + curl 测试 | `/health` 返回 OK, `/ground_gui` 返回坐标 |
| 2. 通信层 | bridge.py ↔ wsl_worker.py | Windows curl → WSL2 返回正确结果 |
| 3. MCP 集成 | mcp_server.py → Claude Code | Claude 截图+定位+点击 全链路通 |
| 4. 完善 | 配置调优 + 容错 + 文档 | 所有边界场景覆盖 |

---

## 配置参考

```yaml
worker:
  model_path: "nvidia/LocateAnything-3B"
  device: "cuda"
  dtype: "bfloat16"

server:
  host: "0.0.0.0"
  port: 8765

inference:
  default_mode: "fast"
  fallback_mode: "hybrid"
  max_new_tokens: 2048
  temperature: 0.3
  top_p: 0.85
  repetition_penalty: 1.05
```

---

## 参考

- [LocateAnything-3B 深度技术分析](../../../tmp/LocateAnything-3B_深度分析.md)
- [LocateAnything Fast 模式深度剖析](../../../tmp/LocateAnything_Fast模式深度剖析.md)
- 论文: arxiv:2605.27365
- 模型: https://huggingface.co/nvidia/LocateAnything-3B
- 代码: https://github.com/NVlabs/Eagle/tree/main/Embodied
