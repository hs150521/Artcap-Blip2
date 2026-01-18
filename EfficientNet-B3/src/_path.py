from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """
    允许直接用：
      python EfficientNet-B3/src/train.py ...
    因为顶层目录名包含 '-'，不作为可导入包名，所以这里显式把 src 加入 sys.path。
    """
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
















