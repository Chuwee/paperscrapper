"""
Telegram Bot interface for Cortex paper recommendation system.
"""
import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

import database

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /start command.
    Welcomes the user and registers them in the database.
    """
    chat_id = update.effective_chat.id
    
    # Add user to database
    database.add_user(chat_id)
    logger.info(f"User {chat_id} registered")
    
    # Send welcome message
    await update.message.reply_text(
        "Welcome to Cortex. Entering Calibration Mode."
    )


def send_paper(chat_id: int, paper_dict: dict):
    """
    Construct and send a paper card with inline keyboard buttons.
    
    Args:
        chat_id: The Telegram chat ID to send the message to
        paper_dict: Dictionary containing paper data with keys:
            - id: Paper ID in database
            - title: Paper title
            - sub_category: Paper category (e.g., 'Math.ST')
            - published: Published date
            - abstract: Paper abstract
            - arxiv_id: ArXiv ID for PDF link
    
    Returns:
        Dictionary of message parameters for sending via Telegram API
    """
    # Extract paper information
    paper_id = paper_dict.get('id')
    title = paper_dict.get('title', 'Untitled')
    category = paper_dict.get('sub_category', 'Unknown')
    published = paper_dict.get('published', '')
    abstract = paper_dict.get('abstract', '')
    arxiv_id = paper_dict.get('arxiv_id', '')
    
    # Format the published date (extract just the date part if it's a full timestamp)
    if published:
        # Handle both string and datetime formats
        if hasattr(published, 'strftime'):
            published_str = published.strftime('%Y-%m-%d')
        else:
            # If it's already a string, just take the first 10 characters (YYYY-MM-DD)
            published_str = str(published)[:10]
    else:
        published_str = 'Unknown'
    
    # Truncate abstract to first 300 characters
    key_abstract = abstract[:300] if len(abstract) > 300 else abstract
    if len(abstract) > 300:
        key_abstract += "..."
    
    # Construct PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    # Format the message with HTML
    message_text = (
        f"<b>{title}</b>\n"
        f"<i>{category} | {published_str}</i>\n\n"
        f"Key Abstract: {key_abstract}\n\n"
        f"<a href='{pdf_url}'>Read Full PDF</a>"
    )
    
    # Create inline keyboard with two buttons
    keyboard = [
        [
            InlineKeyboardButton("✅ Interested", callback_data=f"like_{paper_id}"),
            InlineKeyboardButton("❌ Skip", callback_data=f"dislike_{paper_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Return the coroutine for sending the message
    # Note: This needs to be called with await from an async context
    return {
        'chat_id': chat_id,
        'text': message_text,
        'parse_mode': ParseMode.HTML,
        'reply_markup': reply_markup,
        'disable_web_page_preview': True,
    }


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle callback queries from inline keyboard buttons.
    Logs the interaction and updates the message to show feedback.
    """
    query = update.callback_query
    await query.answer()
    
    # Parse callback data
    callback_data = query.data
    user_id = update.effective_user.id
    
    # Extract action and paper_id
    # Format: "like_{paper_id}" or "dislike_{paper_id}"
    if callback_data.startswith("like_"):
        action = "interested"
        paper_id = int(callback_data.split("_")[1])
        feedback_text = "Recorded: Interested"
    elif callback_data.startswith("dislike_"):
        action = "not_interested"
        paper_id = int(callback_data.split("_")[1])
        feedback_text = "Recorded: Skipped"
    else:
        logger.warning(f"Unknown callback data: {callback_data}")
        return
    
    # Log interaction to database
    database.log_interaction(user_id, paper_id, action)
    logger.info(f"User {user_id} {action} paper {paper_id}")
    
    # Edit the message to append feedback and remove buttons
    original_text = query.message.text_html
    updated_text = f"{original_text}\n\n<b>{feedback_text}</b>"
    
    await query.edit_message_text(
        text=updated_text,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


def main() -> None:
    """
    Main function to set up and run the Telegram bot.
    """
    # Get bot token from environment
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not found in environment variables. "
            "Please create a .env file with your bot token."
        )
    
    # Initialize database
    database.init_db()
    logger.info("Database initialized")
    
    # Create the Application
    application = Application.builder().token(token).build()
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    
    # Register callback query handler for button clicks
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start the bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
