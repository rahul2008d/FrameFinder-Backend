from loguru import logger
import sys
from settings import settings

def setup_logging():
    logger.remove()
    level = "DEBUG" if settings.environment != "prod" else "INFO"
    logger.add(sys.stderr, level=level, backtrace=False, diagnose=False,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                      "<level>{message}</level>")
    return logger
