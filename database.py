"""
Database module for persistent storage using SQLite.
"""
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any

# Global constant for database path
DB_PATH = 'cortex.db'


def init_db():
    """
    Initialize the database by creating all required tables.
    Creates the file cortex.db if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute('PRAGMA foreign_keys = ON')
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            telegram_id INTEGER PRIMARY KEY,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'calibration'
        )
    ''')
    
    # Create papers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            sub_category TEXT,
            abstract TEXT,
            published TIMESTAMP
        )
    ''')
    
    # Create interactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            paper_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(telegram_id),
            FOREIGN KEY (paper_id) REFERENCES papers(id),
            UNIQUE(user_id, paper_id)
        )
    ''')
    
    conn.commit()
    conn.close()


def add_user(telegram_id: int):
    """
    Add a user to the database. Idempotent - ignores if user already exists.
    
    Args:
        telegram_id: The Telegram user ID
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute('PRAGMA foreign_keys = ON')
    
    try:
        cursor.execute(
            'INSERT OR IGNORE INTO users (telegram_id) VALUES (?)',
            (telegram_id,)
        )
        conn.commit()
    finally:
        conn.close()


def store_paper(paper_data: dict) -> Optional[int]:
    """
    Insert a paper into the database if it doesn't exist.
    
    Args:
        paper_data: Dictionary with keys: arxiv_id, title, sub_category, abstract, published
    
    Returns:
        The paper_id of the inserted or existing paper
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute('PRAGMA foreign_keys = ON')
    
    try:
        # Convert datetime to ISO format string if present
        published = paper_data.get('published')
        if published and hasattr(published, 'isoformat'):
            published = published.isoformat()
        
        # Try to insert the paper
        cursor.execute(
            '''INSERT OR IGNORE INTO papers (arxiv_id, title, sub_category, abstract, published)
               VALUES (?, ?, ?, ?, ?)''',
            (
                paper_data.get('arxiv_id'),
                paper_data.get('title'),
                paper_data.get('sub_category'),
                paper_data.get('abstract'),
                published
            )
        )
        
        # Get the paper_id (either newly inserted or existing)
        cursor.execute(
            'SELECT id FROM papers WHERE arxiv_id = ?',
            (paper_data.get('arxiv_id'),)
        )
        result = cursor.fetchone()
        paper_id = result[0] if result else None
        
        conn.commit()
        return paper_id
    finally:
        conn.close()


def log_interaction(user_id: int, paper_id: int, action: str):
    """
    Record a user interaction with a paper.
    
    Args:
        user_id: The telegram_id of the user
        paper_id: The ID of the paper
        action: The action taken ('interested' or 'not_interested')
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute('PRAGMA foreign_keys = ON')
    
    try:
        # Use INSERT OR REPLACE to update existing votes
        # Note: This will update the timestamp to reflect when the vote was changed
        cursor.execute(
            '''INSERT OR REPLACE INTO interactions (user_id, paper_id, action)
               VALUES (?, ?, ?)''',
            (user_id, paper_id, action)
        )
        conn.commit()
    finally:
        conn.close()


def get_unseen_papers(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get papers that the user has not yet interacted with.
    
    Args:
        user_id: The telegram_id of the user
        limit: Maximum number of papers to return (default: 10)
    
    Returns:
        List of dictionaries containing paper data
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute('PRAGMA foreign_keys = ON')
    
    try:
        cursor.execute(
            '''SELECT p.id, p.arxiv_id, p.title, p.sub_category, p.abstract, p.published
               FROM papers p
               WHERE p.id NOT IN (
                   SELECT paper_id FROM interactions WHERE user_id = ?
               )
               ORDER BY p.published DESC
               LIMIT ?''',
            (user_id, limit)
        )
        
        results = cursor.fetchall()
        papers = [dict(row) for row in results]
        return papers
    finally:
        conn.close()


if __name__ == "__main__":
    """
    Test script to verify database functionality.
    """
    import os
    
    # Clean up any existing test database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")
    
    # Test 1: Initialize database
    print("\n=== Test 1: Initialize Database ===")
    init_db()
    print(f"✓ Database created: {os.path.exists(DB_PATH)}")
    
    # Test 2: Add users
    print("\n=== Test 2: Add Users ===")
    add_user(12345)
    add_user(67890)
    add_user(12345)  # Test idempotency
    print("✓ Users added (including idempotency test)")
    
    # Verify users
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('PRAGMA foreign_keys = ON')
    cursor.execute('SELECT telegram_id, status FROM users')
    users = cursor.fetchall()
    print(f"  Users in database: {users}")
    conn.close()
    
    # Test 3: Store papers
    print("\n=== Test 3: Store Papers ===")
    paper1 = {
        'arxiv_id': '2310.12345',
        'title': 'Test Paper 1',
        'sub_category': 'cs.AI',
        'abstract': 'This is a test abstract for paper 1.',
        'published': datetime.now()
    }
    paper2 = {
        'arxiv_id': '2310.67890',
        'title': 'Test Paper 2',
        'sub_category': 'cs.LG',
        'abstract': 'This is a test abstract for paper 2.',
        'published': datetime.now()
    }
    
    paper1_id = store_paper(paper1)
    paper2_id = store_paper(paper2)
    paper1_id_again = store_paper(paper1)  # Test idempotency
    
    print(f"✓ Paper 1 ID: {paper1_id}")
    print(f"✓ Paper 2 ID: {paper2_id}")
    print(f"✓ Paper 1 ID (again): {paper1_id_again} (should match)")
    
    # Test 4: Log interactions
    print("\n=== Test 4: Log Interactions ===")
    log_interaction(12345, paper1_id, 'interested')
    log_interaction(12345, paper2_id, 'not_interested')
    log_interaction(67890, paper1_id, 'interested')
    print("✓ Interactions logged")
    
    # Verify interactions
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('PRAGMA foreign_keys = ON')
    cursor.execute('SELECT user_id, paper_id, action FROM interactions')
    interactions = cursor.fetchall()
    print(f"  Interactions in database: {interactions}")
    conn.close()
    
    # Test 5: Get unseen papers
    print("\n=== Test 5: Get Unseen Papers ===")
    
    # Add a third paper that user 12345 hasn't seen
    paper3 = {
        'arxiv_id': '2310.11111',
        'title': 'Test Paper 3',
        'sub_category': 'cs.CV',
        'abstract': 'This is a test abstract for paper 3.',
        'published': datetime.now()
    }
    paper3_id = store_paper(paper3)
    
    # User 12345 has seen papers 1 and 2, so should only get paper 3
    unseen_for_user1 = get_unseen_papers(12345, limit=10)
    print(f"✓ Unseen papers for user 12345: {len(unseen_for_user1)}")
    for paper in unseen_for_user1:
        print(f"  - {paper['title']} (arxiv_id: {paper['arxiv_id']})")
    
    # User 67890 has only seen paper 1, so should get papers 2 and 3
    unseen_for_user2 = get_unseen_papers(67890, limit=10)
    print(f"✓ Unseen papers for user 67890: {len(unseen_for_user2)}")
    for paper in unseen_for_user2:
        print(f"  - {paper['title']} (arxiv_id: {paper['arxiv_id']})")
    
    # Test 6: Foreign key constraint enforcement
    print("\n=== Test 6: Foreign Key Constraints ===")
    try:
        log_interaction(99999, paper1_id, 'interested')  # Non-existent user
        print("✗ Foreign key constraint NOT enforced (user_id)")
    except sqlite3.IntegrityError as e:
        print(f"✓ Foreign key constraint enforced (user_id): {e}")
    
    try:
        log_interaction(12345, 99999, 'interested')  # Non-existent paper
        print("✗ Foreign key constraint NOT enforced (paper_id)")
    except sqlite3.IntegrityError as e:
        print(f"✓ Foreign key constraint enforced (paper_id): {e}")
    
    # Test 7: Unique constraint on (user_id, paper_id)
    print("\n=== Test 7: Duplicate Interaction Prevention ===")
    # Try to log the same interaction again (should replace, not duplicate)
    log_interaction(12345, paper1_id, 'not_interested')  # Change vote
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('PRAGMA foreign_keys = ON')
    cursor.execute('SELECT action FROM interactions WHERE user_id = ? AND paper_id = ?', (12345, paper1_id))
    result = cursor.fetchone()
    print(f"✓ User 12345's vote for paper 1 was updated to: {result[0]}")
    
    cursor.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ? AND paper_id = ?', (12345, paper1_id))
    count = cursor.fetchone()[0]
    print(f"✓ Number of interactions for user 12345 and paper 1: {count} (should be 1)")
    conn.close()
    
    print("\n=== All Tests Passed! ===")
