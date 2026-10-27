"""pytest 公共配置：把项目根目录塞进 sys.path，避免安装成包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
