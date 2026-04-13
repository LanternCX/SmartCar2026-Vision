"""测试辅助函数."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"


def load_main_module(module_name: str):
    """按真实模块导入方式加载 main.py, 但不触发运行入口."""
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
