import os
import dotenv
import asyncio
import telegram
import telegram.ext

dotenv.load_dotenv()

_TELEGRAM_BOT_API = os.environ['TELEGRAM_BOT_API']

async def _get_me_hf0():
    bot = telegram.Bot(_TELEGRAM_BOT_API)
    async with bot:
        print(await bot.get_me())


def demo_get_me():
    asyncio.run(_get_me_hf0())
