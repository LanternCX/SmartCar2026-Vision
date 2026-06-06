"""ChromaForge 导出文件到 OpenART 配置片段的转换工具。"""

import argparse
import json
from pathlib import Path


_FORMAT = "chromaforge-v1"
_TARGET = "openmv-find-blobs"
DEFAULT_RULES_PATH = Path(__file__).with_name("chromaforge-rules.json")


def _require_document(document):
    if document.get("format") != _FORMAT:
        raise ValueError("ChromaForge 导出格式不匹配")
    if document.get("target") != _TARGET:
        raise ValueError("ChromaForge 导出目标不匹配")
    objects = document.get("objects")
    if not isinstance(objects, list):
        raise ValueError("ChromaForge 导出缺少 objects")


def load_rules_json(path=None):
    """读取并校验 ChromaForge 规则文件。"""

    rules_path = Path(path) if path is not None else DEFAULT_RULES_PATH
    export_json = rules_path.read_text(encoding="utf-8")
    _require_document(json.loads(export_json))
    return export_json


def _threshold_tuple(threshold):
    if not isinstance(threshold, list) or len(threshold) != 6:
        raise ValueError("LAB 阈值必须包含 6 个数字")
    return tuple(int(value) for value in threshold)


def _task_entries(objects):
    entries = []
    for item in objects:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("物体名称不能为空")
        thresholds = [_threshold_tuple(threshold) for threshold in item.get("thresholds", [])]
        if not thresholds:
            continue
        if bool(item.get("require_all_clusters", True)):
            entries.append((name, tuple(thresholds)))
            continue
        for threshold in thresholds:
            entries.append((name, threshold))
    return entries


def _format_threshold(threshold):
    return "(" + ", ".join(str(value) for value in threshold) + ")"


def _format_task(task):
    name, thresholds = task
    if thresholds and isinstance(thresholds[0], tuple):
        inner = ", ".join(_format_threshold(threshold) for threshold in thresholds)
        if len(thresholds) == 1:
            inner += ","
        return "    ('%s', (%s))," % (name, inner)
    return "    ('%s', %s)," % (name, _format_threshold(thresholds))


def build_openart_config(export_json, task_constant_name="TASKS"):
    """把 ChromaForge JSON 转成 OpenART 视觉入口可复制的配置片段。"""

    document = json.loads(export_json)
    _require_document(document)
    entries = _task_entries(document["objects"])
    lines = [
        "# 由 ChromaForge 导出的 OpenART 色块识别参数。",
        "OBJECT_BLOB_MERGE_MARGIN = %d" % int(document.get("recognition_merge_gap", 0)),
        "OBJECT_BLOB_PIXELS_THRESHOLD = %d"
        % int(document.get("min_recognition_block_area", 1)),
        "OBJECT_BLOB_AREA_THRESHOLD = %d"
        % int(document.get("min_recognition_target_area", 1)),
        "%s = (" % task_constant_name,
    ]
    lines.extend(_format_task(entry) for entry in entries)
    lines.append(")")
    return "\n".join(lines) + "\n"


def _find_assignment_line(lines, constant_name):
    prefix = "%s = " % constant_name
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index
    raise ValueError("未找到 %s 配置入口" % constant_name)


def _find_task_block(lines, task_constant_name):
    task_start = None
    for index, line in enumerate(lines):
        if line.startswith("%s = (" % task_constant_name):
            task_start = index
            break
    if task_start is None:
        raise ValueError("未找到 %s 配置入口" % task_constant_name)

    depth = 0
    for index in range(task_start, len(lines)):
        depth += lines[index].count("(")
        depth -= lines[index].count(")")
        if index > task_start and depth <= 0:
            return task_start, index + 1
    raise ValueError("%s 配置块不完整" % task_constant_name)


def build_role_source(source_text, export_json, task_constant_name="TASKS"):
    """把 ChromaForge 配置写入角色 main.py 文本，供构建脚本生成部署文件。"""

    document = json.loads(export_json)
    _require_document(document)
    lines = source_text.splitlines()
    values = {
        "OBJECT_BLOB_MERGE_MARGIN": int(document.get("recognition_merge_gap", 0)),
        "OBJECT_BLOB_PIXELS_THRESHOLD": int(
            document.get("min_recognition_block_area", 1)
        ),
        "OBJECT_BLOB_AREA_THRESHOLD": int(
            document.get("min_recognition_target_area", 1)
        ),
    }
    for name, value in values.items():
        lines[_find_assignment_line(lines, name)] = "%s = %d" % (name, value)

    start, end = _find_task_block(lines, task_constant_name)
    task_lines = ["%s = (" % task_constant_name]
    task_lines.extend(_format_task(entry) for entry in _task_entries(document["objects"]))
    task_lines.append(")")
    return "\n".join(lines[:start] + task_lines + lines[end:]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="把 ChromaForge 导出的 JSON 转成 OpenART 配置入口"
    )
    parser.add_argument(
        "export_json",
        nargs="?",
        help="ChromaForge 导出的 chromaforge-rules.json，默认读取工具同目录文件",
    )
    parser.add_argument(
        "--task-constant-name",
        default="TASKS",
        help="输出的任务常量名，例如 TASKS、OBJECT_TASKS 或 FOLLOW_TASKS",
    )
    parser.add_argument("--source", help="角色 main.py 源文件")
    parser.add_argument("--output", help="写入适配后的角色 main.py")
    args = parser.parse_args(argv)
    export_json = load_rules_json(args.export_json)
    if args.source:
        with open(args.source, "r", encoding="utf-8") as file:
            result = build_role_source(
                file.read(), export_json, task_constant_name=args.task_constant_name
            )
    else:
        result = build_openart_config(
            export_json, task_constant_name=args.task_constant_name
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
