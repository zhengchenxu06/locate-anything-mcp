"""
WSL2 侧 FastAPI Worker — 常驻 LocateAnything-3B 模型
启动: cd /mnt/c/Users/Administrator/locate-anything-mcp && python3 wsl_worker.py
"""
import io, base64, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

app = FastAPI(title="LocateAnything Worker")
_worker = None


def _load_worker():
    global _worker
    if _worker is None:
        model_path = "/home/locate-anything-model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        _worker = LocateAnythingWorker(model_path, device=device, dtype=dtype)
    return _worker


class LocateAnythingWorker:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        use_int4 = (device == "cuda")
        if use_int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
            ).eval()
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True,
            ).to(device).eval()
        self.tokenizer = self.processor.tokenizer

    @torch.no_grad()
    def predict(self, image, question, generation_mode="fast",
                max_new_tokens=2048, temperature=0.3):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, _ = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.05,
        )
        return response[0] if isinstance(response, tuple) else response


class GroundGUIRequest(BaseModel):
    image_b64: str
    description: str
    mode: str = "fast"


@app.get("/health")
def health():
    gpu_ok = torch.cuda.is_available()
    return {
        "status": "ok",
        "model_loaded": _worker is not None,
        "gpu_available": gpu_ok,
        "vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        ) if gpu_ok else 0,
    }


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
        return {"raw_answer": result, "boxes": _parse_boxes(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        return {
            "raw_answer": result,
            "boxes_by_category": _parse_boxes_grouped(result, categories),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _parse_boxes(answer: str) -> list:
    import re
    boxes = []
    # 匹配 <box><x1><y1><x2><y2></box>
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        boxes.append({"bbox": [int(x) for x in m.groups()], "score": 1.0})
    # 匹配 <box><x><y></box> (点定位)
    if not boxes:
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            boxes.append({"bbox": [x, y, x, y], "score": 1.0})
    return boxes


def _parse_boxes_grouped(answer: str, categories: list) -> dict:
    boxes = _parse_boxes(answer)
    result = {cat: [] for cat in categories}
    for i, box in enumerate(boxes):
        cat = categories[i % len(categories)]
        result[cat].append(box)
    return result


if __name__ == "__main__":
    import uvicorn
    _load_worker()  # 启动时预加载
    uvicorn.run(app, host="0.0.0.0", port=8765)
