# paperscrapper

A Telegram bot for discovering and curating academic papers from arXiv.

## Features

- **Paper Harvesting**: Fetches latest papers from arXiv by category
- **User Calibration**: Presents papers to users for feedback
- **Interaction Tracking**: Records user preferences for future recommendations
- **Telegram Interface**: Easy-to-use bot interface with inline buttons

## Setup

### Prerequisites

- Python 3.7+
- Telegram Bot Token (get one from [@BotFather](https://t.me/botfather))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Chuwee/paperscrapper.git
cd paperscrapper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your bot token:
```bash
cp .env.example .env
# Edit .env and add your TELEGRAM_BOT_TOKEN
```

### Running the Bot

```bash
python bot.py
```

## Usage

1. Start a chat with your bot on Telegram
2. Send `/start` to begin calibration mode
3. The bot will present papers with:
   - Title and category
   - Publication date
   - Abstract preview
   - Link to full PDF
4. Click ✅ Interested or ❌ Skip to provide feedback

## Modules

- `database.py`: SQLite database for storing papers, users, and interactions
- `harvester.py`: arXiv paper fetching functionality
- `bot.py`: Telegram bot interface

## Testing

Run the test suite:
```bash
python test_bot.py
```

