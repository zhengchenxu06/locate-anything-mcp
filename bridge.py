"""Windows ↔ WSL2 HTTP 通信桥"""
import base64, io, requests
from PIL import Image
from typing import Optional


class LocateAnythingBridge:
    def __init__(self, base_url: str = "http://localhost:8765", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def ground_gui(self, image_b64: str, description: str,
                   mode: str = "fast") -> dict:
        resp = requests.post(
            f"{self.base_url}/ground_gui",
            json={"image_b64": image_b64, "description": description, "mode": mode},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def locate_all(self, image_b64: str, categories: list[str],
                   mode: str = "fast") -> dict:
        resp = requests.post(
            f"{self.base_url}/locate_all",
            json={"image_b64": image_b64, "categories": categories, "mode": mode},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


def pil_to_b64(image: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode()


def resize_if_needed(image: Image.Image, max_dim: int = 2500) -> Image.Image:
    w, h = image.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return image
