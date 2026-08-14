import os
import sys
from zoneinfo import ZoneInfo

from loguru import logger

from src.config import settings

# Timestamps are stamped in US Eastern rather than the host clock: the training
# runs happen in containers whose clock is UTC, so a log line's time did not
# line up with the wall clock anyone was reading it against. ZoneInfo (not a
# fixed -05:00 offset) so DST is handled.
EASTERN = ZoneInfo("America/New_York")


def _to_eastern(record) -> None:
    record["time"] = record["time"].astimezone(EASTERN)


logger.configure(patcher=_to_eastern)

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss zz}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()
logger.level("DEBUG", color="<cyan>")
logger.level("INFO", color="<white>")
logger.level("SUCCESS", color="<green>")
logger.level("WARNING", color="<yellow><bold>")
logger.level("ERROR", color="<red><bold>")
logger.level("CRITICAL", color="<red><bold><reverse>")

# The console sink drops the module:function:line prefix that the file sink
# keeps: under `docker logs` it pushed the episode lines past the terminal width
# and wrapped every one of them.
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss zz}</green> | "
    "<level>{level: <7}</level> | "
    "<level>{message}</level>"
)

logger.add(sys.stderr, format=CONSOLE_FORMAT, level="INFO", colorize=True)
logger.add(
    os.path.join(settings.LOG_DIR, f"{settings.RUN_ID}.log"),
    format=LOG_FORMAT,
    level="DEBUG",
    colorize=False,
    rotation="10 MB",
    retention=5,
    enqueue=True,
)

__all__ = ["logger"]
