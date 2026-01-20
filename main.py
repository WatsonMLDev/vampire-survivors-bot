import warnings
# Suppress pkg_resources warning from pygame and others
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import dotenv
from bot.system.logger import logger
from bot.core.bot import VampireSurvivorsBot

dotenv.load_dotenv()

def main():
    logger.info("Starting main execution...")
    bot = None
    try:
        bot = VampireSurvivorsBot()
        bot.run()
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in main: {e}")
        import traceback
        traceback.print_exc()
        if bot:
            bot.stop()

if __name__ == "__main__":
    main()
