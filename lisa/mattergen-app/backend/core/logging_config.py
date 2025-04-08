import sys
from pathlib import Path
from loguru import logger
from datetime import timedelta 
import json

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure loguru logger
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {extra} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "level": "DEBUG",
        },
        {
            "sink": "logs/info.log",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra} | {name}:{function}:{line} - {message}",
            "level": "INFO",
            "rotation": "1 day",
            "retention": timedelta(days=30),
            "compression": "zip",
        },
        {
            "sink": "logs/error.log",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra} | {name}:{function}:{line} - {message}",
            "level": "ERROR",
            "rotation": "1 day",
            "retention": timedelta(days=30),
            "compression": "zip",
        },
        {
            "sink": "logs/debug.log",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra} | {name}:{function}:{line} - {message}",
            "level": "DEBUG",
            "rotation": "1 day",
            "retention": timedelta(days=30),
            "compression": "zip",
        },
    ],
}

# Remove default logger
logger.remove()

# Apply configuration
for handler in config["handlers"]:
    logger.add(**handler)

# Export logger
get_logger = logger.bind