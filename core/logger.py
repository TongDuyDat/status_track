from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
import logging
import sys
import os

# Get log level from env
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a")  # ✅ Log to file
    ]
)

logger = logging.getLogger("pipeline")
logger.setLevel(getattr(logging, LOG_LEVEL))