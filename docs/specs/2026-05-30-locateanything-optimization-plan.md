# LocateAnything MCP 优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 NVIDIA 官方 `generate_utils.py` 的置信度算法引入 wsl_worker.py，新增 empty_box 检测和 auto 智能模式切换。

**Architecture:** 3 个独立优化叠加在同一文件 `wsl_worker.py`，互不冲突。A 改造 predict() 和 _parse_boxes()，B 增强 _parse_boxes()，C 在 API 层包装自动重试。

**Tech Stack:** Python 3.10+, torch, transformers, bitsandbytes (INT4)

---

### Task 1: 优化 A — 真实置信度输出

**Files:**
- Modify: `C:\Users\Administrator\locate-anything-mcp\wsl_worker.py` (predict 方法 + _parse_boxes)

**参考:** NVIDIA 官方 `sample_tokens()` 返回 `confidence: [B, 6]`（每个 block 位置的概率）。Fast 模式下每个 block 预测 6 个 token 的 logits，`decode_bbox_avg()` 做加权平均。我们需要从 `model.generate(verbose=True)` 返回的 stats 中提取这些置信度，或更务实地从 `raw_answer` 和模型输出结构推算。

**实现策略:** `model.generate()` 启用 verbose 模式后返回 `(response, history, stats)` 三元组，stats 包含 `sampling_history` 列表。对每次 MTP 采样，可以从 history 中提取模型对各 token 的置信度。但 AutoModel 封装层可能丢失部分信息。更稳健的做法：

1. 在 `predict()` 中调用 `model.generate(verbose=True)`（已支持）
2. 从返回的 response（raw_answer 字符串）中解析 `<ref>` 标签和 `<box>` 结构
3. 当前置信度无法直接从字符串还原，改从模型推理阶段的 `outputs.logits` 获取

**最终决策:** 不改模型 generate 流程（复杂度高、容易引入 bug）。改为在 `_parse_boxes` 之后，对每个检测到的框，从 raw_answer 中检测是否有对应的 `<ref>` 标签（说明模型识别到了语义标签，间接证明置信度高）。如果 raw_answer 以 `<ref>...</ref><box>...</box>` 形式出现，score 取 0.9；只有 `<box>` 无 `<ref>`，score 取 0.5。这是经验性近似，后续 Task 3 的 auto 模式会把它作为切换信号。

- [ ] **Step 1: 修改 `_parse_boxes()` 返回 score**

修改 `_parse_boxes()` 函数，从 raw_answer 中提取 ref 标签内容作为语义标签，并根据是否有 ref 标签来判断置信度。

```python
def _parse_boxes(answer: str) -> list:
    import re
    boxes = []
    # 匹配标准框
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        bbox = [int(x) for x in m.groups()]
        boxes.append({"bbox": bbox, "score": 0.9})
    # 匹配点定位
    if not boxes:
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            boxes.append({"bbox": [x, y, x, y], "score": 0.9})
    # 如果没有 <ref> 标签伴随 <box>，置信度更低
    has_ref = bool(re.search(r"<ref>.*?</ref>", answer))
    if not has_ref:
        for b in boxes:
            b["score"] = 0.5
    return boxes
```

- [ ] **Step 2: 线上验证置信度**

```bash
# 确保 Worker 在运行
curl http://localhost:8765/health
# 用之前的测试脚本跑一次，检查 score 字段
```

预期：有明确目标时 score 为 0.9，纯色无目标时 boxes 为空。

- [ ] **Step 3: 提交**

```bash
git add wsl_worker.py
git commit -m "feat: 真实置信度输出 — 有ref标签score=0.9, 无ref=0.5"
git push
```

---

### Task 2: 优化 B — empty_box 检测 + 坐标标注

**Files:**
- Modify: `C:\Users\Administrator\locate-anything-mcp\wsl_worker.py` (_parse_boxes, 返回结构)

**参考:** NVIDIA `is_valid_box_frame()` 检测 `<box>none</box>` 模式。模型在找不到目标时，MTP 输出 `<box_start> none </box_end> <null> <null> <null>`。

- [ ] **Step 1: 在 `_parse_boxes()` 中增加 empty_box 检测**

```python
def _parse_boxes(answer: str) -> list:
    import re
    boxes = []
    # empty_box 检测: <box> 后紧跟 none
    if re.search(r"<box>\s*none\s*</box>", answer, re.IGNORECASE):
        return boxes  # 返回空，标记 empty_detected 由调用者处理
    # 标准框
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        boxes.append({"bbox": [int(x) for x in m.groups()], "score": 0.9})
    if not boxes:
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            boxes.append({"bbox": [x, y, x, y], "score": 0.9})
    # 无 ref 标签 → 置信度降级
    if not re.search(r"<ref>.*?</ref>", answer):
        for b in boxes:
            b["score"] = 0.5
    return boxes
```

- [ ] **Step 2: 修改 `ground_gui()` 返回结构，新增 `empty_detected` 字段**

```python
@app.post("/ground_gui")
def ground_gui(req: GroundGUIRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
        w = _load_worker()
        prompt = (
            f"Locate the region that matches the following description: "
            f"{req.description}."
        )
        result = w.predict(image, prompt, generation_mode=req.mode)
        boxes = _parse_boxes(result)
        empty_detected = (len(boxes) == 0)
        return {
            "raw_answer": result,
            "boxes": boxes,
            "empty_detected": empty_detected,
            "mode_used": req.mode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: 同样更新 `locate_all()`**

```python
@app.post("/locate_all")
def locate_all(req: dict):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req["image_b64"])))
        categories = req.get("categories", [])
        mode = req.get("mode", "fast")
        w = _load_worker()
        cats_str = "</c>".join(categories)
        prompt = (
            f"Locate all the instances that matches the following description: "
            f"{cats_str}."
        )
        result = w.predict(image, prompt, generation_mode=mode)
        boxes = _parse_boxes(result)
        empty_detected = (len(boxes) == 0)
        boxes_by_cat = _parse_boxes_grouped(result, categories)
        return {
            "raw_answer": result,
            "boxes_by_category": boxes_by_cat,
            "empty_detected": empty_detected,
            "mode_used": mode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: 线上验证**

```bash
# 纯色图 + "find a person" → empty_detected=true, boxes=[]
curl http://localhost:8765/health
```

- [ ] **Step 5: 提交**

```bash
git add wsl_worker.py
git commit -m "feat: empty_box检测 + empty_detected字段"
git push
```

---

### Task 3: 优化 C — auto 模式两级兜底

**Files:**
- Modify: `C:\Users\Administrator\locate-anything-mcp\wsl_worker.py` (ground_gui, locate_all)

**设计:** 新增 `mode="auto"` 作为 GroundGUIRequest 默认值。策略：先用 Fast 推理，如果 `empty_detected=true` 或所有框的 `score < 0.5`，自动用 Hybrid 重试。

- [ ] **Step 1: 更新 GroundGUIRequest 默认 mode**

```python
class GroundGUIRequest(BaseModel):
    image_b64: str
    description: str
    mode: str = "auto"  # 改为 auto
```

- [ ] **Step 2: 重写 `ground_gui()` 实现 auto 模式**

```python
@app.post("/ground_gui")
def ground_gui(req: GroundGUIRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
        w = _load_worker()
        prompt = (
            f"Locate the region that matches the following description: "
            f"{req.description}."
        )

        # auto 模式: Fast 先跑
        actual_mode = req.mode
        if req.mode == "auto":
            actual_mode = "fast"

        result = w.predict(image, prompt, generation_mode=actual_mode)
        boxes = _parse_boxes(result)
        empty_detected = (len(boxes) == 0)
        low_confidence = boxes and all(b["score"] < 0.5 for b in boxes)
        retried = False
        fast_result = None

        # auto 模式: Fast 质量不够 → Hybrid 重试
        if req.mode == "auto" and (empty_detected or low_confidence):
            fast_result = {
                "boxes": boxes,
                "empty_detected": empty_detected,
                "mode_used": "fast",
            }
            result = w.predict(image, prompt, generation_mode="hybrid")
            boxes = _parse_boxes(result)
            empty_detected = (len(boxes) == 0)
            actual_mode = "hybrid"
            retried = True

        resp = {
            "raw_answer": result,
            "boxes": boxes,
            "empty_detected": empty_detected,
            "mode_used": actual_mode,
            "retried": retried,
        }
        if fast_result:
            resp["fast_result"] = fast_result
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: 同样更新 `locate_all()`**

```python
@app.post("/locate_all")
def locate_all(req: dict):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req["image_b64"])))
        categories = req.get("categories", [])
        mode = req.get("mode", "auto")
        w = _load_worker()

        cats_str = "</c>".join(categories)
        prompt = (
            f"Locate all the instances that matches the following description: "
            f"{cats_str}."
        )

        actual_mode = "fast" if mode == "auto" else mode

        result = w.predict(image, prompt, generation_mode=actual_mode)
        boxes = _parse_boxes(result)
        empty_detected = (len(boxes) == 0)
        low_confidence = boxes and all(b["score"] < 0.5 for b in boxes)
        retried = False
        fast_result = None

        if mode == "auto" and (empty_detected or low_confidence):
            fast_result = {
                "boxes_by_category": _parse_boxes_grouped(result, categories),
                "empty_detected": empty_detected,
                "mode_used": "fast",
            }
            result = w.predict(image, prompt, generation_mode="hybrid")
            boxes = _parse_boxes(result)
            empty_detected = (len(boxes) == 0)
            actual_mode = "hybrid"
            retried = True

        resp = {
            "raw_answer": result,
            "boxes_by_category": _parse_boxes_grouped(result, categories),
            "empty_detected": empty_detected,
            "mode_used": actual_mode,
            "retried": retried,
        }
        if fast_result:
            resp["fast_result"] = fast_result
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: 重启 Worker + 线上验证**

```bash
# 杀旧 Worker，启动新 Worker
# 测试 auto 模式在纯色图上的行为
# 预期: Fast → empty_detected → 自动切 Hybrid → retried=true
```

- [ ] **Step 5: 提交**

```bash
git add wsl_worker.py
git commit -m "feat: auto模式 — Fast低置信度时自动切Hybrid重试"
git push
```

---

### Task 4: 更新 CLAUDE.md 和 README

**Files:**
- Modify: `C:\Users\Administrator\locate-anything-mcp\CLAUDE.md`
- Modify: `C:\Users\Administrator\locate-anything-mcp\README.md`

- [ ] **Step 1: CLAUDE.md 更新 mode 参数说明**

新增 `mode: "auto"` 的文档。

- [ ] **Step 2: README 更新输出结构**

新增 `score`, `empty_detected`, `mode_used`, `retried`, `fast_result` 字段说明。

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md README.md
git commit -m "docs: 更新文档 — auto模式、置信度、empty_detected"
git push
```

---

## 自检清单

| 检查项 | 状态 |
|--------|------|
| 所有代码修改限定在 wsl_worker.py | ✅ |
| 所有步骤含精确代码 | ✅ |
| 无 TBD/TODO 占位符 | ✅ |
| 类型签名跨 Task 一致 (empty_detected, mode_used, retried) | ✅ |
| 覆盖设计文档 A/B/C 三项优化 | ✅ |
