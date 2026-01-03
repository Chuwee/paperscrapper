"""
Test script for bot.py functionality.
Tests the send_paper function to ensure proper formatting.
"""
import os
import sys
from datetime import datetime

# Import bot module
import bot
import database

def test_send_paper_formatting():
    """Test that send_paper creates the correct message format."""
    print("=== Testing send_paper function ===\n")
    
    # Sample paper data
    test_paper = {
        'id': 1,
        'title': 'Attention Is All You Need',
        'sub_category': 'cs.CL',
        'published': '2017-06-12',
        'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
        'arxiv_id': '1706.03762'
    }
    
    # Generate message parameters
    message_params = bot.send_paper(chat_id=12345, paper_dict=test_paper)
    
    print("Generated message:")
    print("-" * 80)
    print(message_params['text'])
    print("-" * 80)
    print()
    
    # Verify message structure
    message_text = message_params['text']
    
    checks = [
        ("Title in bold", "<b>Attention Is All You Need</b>" in message_text),
        ("Category and date", "<i>cs.CL | 2017-06-12</i>" in message_text),
        ("Abstract truncation", len(message_text.split("Key Abstract:")[1].split("\n\n")[0].strip()) <= 303),  # 300 chars + "..."
        ("PDF link", "https://arxiv.org/pdf/1706.03762.pdf" in message_text),
        ("HTML parse mode", message_params['parse_mode'] == 'HTML'),
        ("Inline keyboard exists", message_params['reply_markup'] is not None),
    ]
    
    print("Verification checks:")
    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False
    
    # Verify keyboard structure
    keyboard = message_params['reply_markup'].inline_keyboard
    print(f"\n  ✓ Keyboard has {len(keyboard)} row(s)")
    print(f"  ✓ First row has {len(keyboard[0])} button(s)")
    
    for i, button in enumerate(keyboard[0]):
        print(f"    - Button {i+1}: '{button.text}' (callback: {button.callback_data})")
        if i == 0:
            assert button.callback_data == "like_1", "First button should have callback 'like_1'"
        elif i == 1:
            assert button.callback_data == "dislike_1", "Second button should have callback 'dislike_1'"
    
    if all_passed:
        print("\n✓ All formatting tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


def test_database_integration():
    """Test that the bot can interact with the database."""
    print("\n=== Testing database integration ===\n")
    
    # Initialize database
    if os.path.exists('cortex.db'):
        os.remove('cortex.db')
    
    database.init_db()
    print("✓ Database initialized")
    
    # Add a test user
    database.add_user(12345)
    print("✓ Test user added")
    
    # Add a test paper
    paper_data = {
        'arxiv_id': '1706.03762',
        'title': 'Attention Is All You Need',
        'sub_category': 'cs.CL',
        'abstract': 'Test abstract',
        'published': datetime.now()
    }
    paper_id = database.store_paper(paper_data)
    print(f"✓ Test paper stored with ID: {paper_id}")
    
    # Log an interaction
    database.log_interaction(12345, paper_id, 'interested')
    print("✓ Interaction logged")
    
    print("\n✓ Database integration works correctly!")


if __name__ == '__main__':
    test_send_paper_formatting()
    test_database_integration()
    print("\n=== All tests passed! ===")
