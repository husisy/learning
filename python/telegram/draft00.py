import os
import logging
import dotenv

import telegram
import telegram.ext

dotenv.load_dotenv()

_TELEGRAM_BOT_API = os.environ['TELEGRAM_BOT_API']

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def hello(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')

async def start(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def echo(update: telegram.Update, context: telegram.ext.CallbackContext):
    tmp0 = f'[echo] {update.message.text}'
    # ' '.join(context.args)
    # await update.message.reply_text(tmp0)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=tmp0)

# async def inline_caps(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
#     query = update.inline_query.query
#     if not query:
#         return
#     results = []
#     results.append(
#         telegram.InlineQueryResultArticle(
#             id=query.upper(),
#             title='Caps',
#             input_message_content=telegram.InputTextMessageContent(query.upper())
#         )
#     )
#     await context.bot.answer_inline_query(update.inline_query.id, results)

async def unknown(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

if __name__ == '__main__':
    app = telegram.ext.ApplicationBuilder().token(_TELEGRAM_BOT_API).build()

    app.add_handler(telegram.ext.CommandHandler("hello", hello))

    app.add_handler(telegram.ext.CommandHandler("start", start))

    # TODO inline mode fail
    # app.add_handler(telegram.ext.InlineQueryHandler(inline_caps))

    app.add_handler(telegram.ext.MessageHandler(telegram.ext.filters.TEXT & (~telegram.ext.filters.COMMAND), echo))

    # Other handlers, MUST be last
    app.add_handler(telegram.ext.MessageHandler(telegram.ext.filters.COMMAND, unknown))

    app.run_polling()
