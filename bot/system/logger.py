import logging
import sys
from bot.system.config import config

def setup_logger(name="VS_Bot"):
    logger = logging.getLogger(name)
    
    # Default to INFO if not specified
    log_level = config.get("logging.level", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    
    logger.setLevel(level)
    
    # Check if handler already exists to verify avoiding duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.propagate = False
    return logger

logger = setup_logger()
