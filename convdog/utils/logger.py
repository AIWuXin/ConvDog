import logging
import os
import re
from datetime import datetime
import onnxruntime as ort

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# 1. 定义主题
custom_theme = Theme({
    "logging.level.success": "green bold",
    "logging.level.info": "cyan",
    "logging.level.warning": "yellow",
    "logging.level.error": "red bold",
    "logging.level.debug": "grey50",
    "success": "green bold",
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "debug": "grey50"
})

# 用于文件日志去色的辅助 Console
_strip_console = Console(width=255, color_system=None)

class RichStripFormatter(logging.Formatter):
    """去除 Rich 标记后写入文件的格式化器"""
    def format(self, record):
        raw_msg = record.msg
        if isinstance(raw_msg, str) and "[" in raw_msg and "]" in raw_msg:
            with _strip_console.capture() as capture:
                _strip_console.print(raw_msg, end="")
            record.msg = capture.get().strip()
        result = super().format(record)
        record.msg = raw_msg
        return result


class WarningFilter(logging.Filter):
    """自定义警告过滤器"""

    def __init__(self):
        super().__init__()
        # 定义要屏蔽的模式
        self.patterns_to_suppress = [
            r'UnsqueezeElimination cannot remove node',
            r'onnxruntime::UnsqueezeElimination::Apply',
            # 可以添加其他要屏蔽的警告模式
        ]

    def filter(self, record):
        # 检查是否匹配要屏蔽的模式
        for pattern in self.patterns_to_suppress:
            if re.search(pattern, record.getMessage()):
                return False  # 不记录这个日志
        return True  # 记录这个日志


class OrtWarningPrintFilter(object):
    def __enter__(self):
        ort.set_default_logger_severity(3)

    def __exit__(self, exc_type, exc_val, exc_tb):
        ort.set_default_logger_severity(2)


class ConvDogLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConvDogLogger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"convdog_{datetime.now().strftime('%Y%m%d')}.log")
        LOG_PREFIX = "[ConvDog🐕]"

        self.logger = logging.getLogger("ConvDog")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        # --- 1. 终端 Handler (展示路径和代码行) ---
        console = Console(theme=custom_theme)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,      # [关键点1] 开启 Rich 侧边显示路径
            enable_link_path=True, # 终端点击路径可跳转代码
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%X]"
        )
        # 终端格式：[ConvDog] 消息
        rich_handler.setFormatter(logging.Formatter(f"[bold list]{LOG_PREFIX}[/] %(message)s"))
        rich_handler.setLevel(logging.INFO)
        self.logger.addHandler(rich_handler)

        # --- 2. 文件 Handler (包含文件名:行号) ---
        # [关键点2] 在字符串中添加 [%(filename)s:%(lineno)d]
        file_formatter = RichStripFormatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        ort_logger = logging.getLogger('onnxruntime')
        warning_filter = WarningFilter()
        ort_logger.addFilter(warning_filter)


    def get_logger(self):
        return self.logger

# 初始化单例
logger = ConvDogLogger().get_logger()

# --- 自定义 SUCCESS 等级注入 ---
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    """
    [关键点3] stacklevel=2 确保识别调用此方法的行号，而不是此处 logger.py 内部行号
    """
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        if isinstance(message, str):
            message = f"[success]{message}[/]"
        # 注入 stacklevel=2
        kws.setdefault("stacklevel", 2)
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

def info_with_color(self, message, *args, **kws):
    if self.isEnabledFor(logging.INFO):
        if isinstance(message, str):
            message = f"[info]{message}[/]"
        # 注入 stacklevel=2
        kws.setdefault("stacklevel", 2)
        self._log(logging.INFO, message, args, **kws)

# 将方法挂载到标准 Logger 类
logging.Logger.success = success
logging.Logger.info = info_with_color
