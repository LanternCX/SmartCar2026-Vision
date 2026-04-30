"""测试辅助函数."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_MAIN_PATH = ROOT / "assistant" / "main.py"


def role_main_path(role: str) -> Path:
    """返回指定角色的视觉入口路径."""

    return ROOT / role / "main.py"


def load_main_module(module_name: str):
    """按真实模块导入方式加载辅车 main.py, 但不触发运行入口."""

    return load_role_main_module("assistant", module_name)


def load_role_main_module(role: str, module_name: str):
    """按真实模块导入方式加载指定角色 main.py, 但不触发运行入口."""

    spec = importlib.util.spec_from_file_location(module_name, role_main_path(role))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
