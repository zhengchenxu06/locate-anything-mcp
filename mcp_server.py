"""LocateAnything MCP Server — 2 工具: ground_gui, locate_all"""
from mcp.server.fastmcp import FastMCP
from bridge import LocateAnythingBridge

mcp = FastMCP("locate-anything")
bridge = LocateAnythingBridge()


@mcp.tool()
def ground_gui(image_b64: str, description: str,
               mode: str = "fast") -> dict:
    """
    定位 GUI 元素。输入截图(base64编码)和自然语言描述，返回元素边界框坐标。

    参数:
        image_b64: PNG/JPG 截图的 base64 编码字符串
        description: 目标元素的自然语言描述，如 "蓝色提交按钮"
        mode: "fast"(默认, 低延迟) 或 "hybrid"(高精度)
    返回:
        {"boxes": [{"bbox": [x1,y1,x2,y2], "score": 0.95}], "raw_answer": "..."}
    """
    return bridge.ground_gui(image_b64, description, mode)


@mcp.tool()
def locate_all(image_b64: str, categories: list[str],
               mode: str = "fast") -> dict:
    """
    检测图像中所有匹配类别的元素。返回按类别分组的边界框。

    参数:
        image_b64: 截图的 base64 编码
        categories: 类别列表，如 ["按钮", "输入框", "图标"]
        mode: "fast" 或 "hybrid"
    返回:
        {"boxes_by_category": {"按钮": [...], "输入框": [...]}}
    """
    return bridge.locate_all(image_b64, categories, mode)


if __name__ == "__main__":
    mcp.run()
