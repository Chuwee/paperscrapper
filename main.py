"""
Main scheduler script that runs the entire Cortex pipeline periodically.

This script:
1. Initializes the database
2. Starts the Telegram bot with polling
3. Uses JobQueue to schedule daily briefing tasks
"""
import os
import logging
import asyncio
from datetime import time
from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
)

import database
import harvester
import manager
import bot as bot_module

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default categories to fetch from arXiv
DEFAULT_CATEGORIES = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'stat.ML']
MAX_PAPERS_PER_FETCH = 50


async def daily_briefing_job(context):
    """
    Daily briefing job that:
    1. Fetches new papers from arXiv
    2. Saves them to the database
    3. Gets all active users
    4. For each user, gets recommendations and sends papers
    
    Args:
        context: The context object from JobQueue
    """
    logger.info("Starting daily briefing job...")
    
    # Step 1: Fetch papers from arXiv
    logger.info(f"Fetching papers from categories: {DEFAULT_CATEGORIES}")
    try:
        papers = harvester.fetch_papers(
            categories=DEFAULT_CATEGORIES,
            max_results=MAX_PAPERS_PER_FETCH
        )
        logger.info(f"Fetched {len(papers)} papers from arXiv")
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        return
    
    # Step 2: Save papers to database
    saved_count = 0
    for paper in papers:
        try:
            # Extract the first category as sub_category, handling empty lists
            categories = paper.get('categories', [])
            sub_category = categories[0] if categories else 'unknown'
            
            # Create paper data dict for database
            paper_data = {
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'sub_category': sub_category,
                'abstract': paper['abstract'],
                'published': paper['published']
            }
            
            paper_id = database.store_paper(paper_data)
            if paper_id:
                saved_count += 1
        except Exception as e:
            logger.error(f"Error saving paper {paper.get('arxiv_id', 'unknown')}: {e}")
    
    logger.info(f"Saved {saved_count} new papers to database")
    
    # Step 3: Get active users
    active_users = database.get_active_users()
    logger.info(f"Found {len(active_users)} active users")
    
    # Step 4: Process each user
    for user in active_users:
        user_id = user['telegram_id']
        logger.info(f"Processing user {user_id}")
        
        try:
            # Get recommendations for this user
            recommendations = manager.get_matches_for_user(user_id)
            logger.info(f"Found {len(recommendations)} recommendations for user {user_id}")
            
            # Send each recommended paper to the user
            for paper in recommendations:
                try:
                    # Create message parameters using bot module's send_paper function
                    message_params = bot_module.send_paper(user_id, paper)
                    
                    # Send the message using the bot from context
                    await context.bot.send_message(**message_params)
                    logger.info(f"Sent paper {paper.get('arxiv_id', 'unknown')} to user {user_id}")
                    
                    # Rate limiting: sleep 1 second between messages
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error sending paper to user {user_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}")
    
    logger.info("Daily briefing job completed")


def main():
    """
    Main function to set up and run the Telegram bot with scheduled jobs.
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
    
    # Register command handlers from bot module
    application.add_handler(CommandHandler("start", bot_module.start))
    
    # Register callback query handler for button clicks
    application.add_handler(CallbackQueryHandler(bot_module.button_callback))
    
    # Get the job queue
    job_queue = application.job_queue
    
    # Schedule the daily briefing job
    # Default: run every 24 hours
    # To run at a specific time (e.g., 08:00 UTC), use:
    #   job_queue.run_daily(daily_briefing_job, time=time(hour=8, minute=0), name='daily_briefing')
    # To test with short interval (e.g., every 10 seconds), use:
    #   job_queue.run_repeating(daily_briefing_job, interval=10, first=5, name='daily_briefing_test')
    
    # Get initial delay from environment or use default of 10 seconds
    initial_delay = int(os.getenv('BRIEFING_INITIAL_DELAY', '10'))
    
    job_queue.run_repeating(
        daily_briefing_job,
        interval=86400,  # 24 hours in seconds
        first=initial_delay,
        name='daily_briefing'
    )
    
    logger.info("Scheduled daily briefing job")
    logger.info("Starting bot polling...")
    
    # Start the bot
    application.run_polling(allowed_updates=['message', 'callback_query'])


if __name__ == '__main__':
    main()
