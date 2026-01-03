"""
Unit tests for manager module.
"""
import os
import sys
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import database
import manager
from brain import CHROMA_DB_PATH


def setup_test_environment():
    """Set up a clean test environment."""
    # Clean up test databases
    if os.path.exists(database.DB_PATH):
        os.remove(database.DB_PATH)
    
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Initialize database
    database.init_db()


def test_database_helper_functions():
    """Test the new database helper functions."""
    print("\n=== Test 1: Database Helper Functions ===")
    setup_test_environment()
    
    # Add a user
    test_user_id = 12345
    database.add_user(test_user_id)
    
    # Test get_user_status
    status = database.get_user_status(test_user_id)
    assert status == 'calibration', f"Expected 'calibration', got {status}"
    print("✓ get_user_status works correctly")
    
    # Test update_user_status
    database.update_user_status(test_user_id, 'active')
    status = database.get_user_status(test_user_id)
    assert status == 'active', f"Expected 'active', got {status}"
    print("✓ update_user_status works correctly")
    
    # Test get_interaction_count (should be 0 initially)
    count = database.get_interaction_count(test_user_id)
    assert count == 0, f"Expected 0 interactions, got {count}"
    print("✓ get_interaction_count works correctly (initial)")
    
    # Add some interactions
    paper1 = {
        'arxiv_id': '2310.00001',
        'title': 'Test Paper 1',
        'sub_category': 'cs.AI',
        'abstract': 'Test abstract 1.',
        'published': datetime.now()
    }
    paper2 = {
        'arxiv_id': '2310.00002',
        'title': 'Test Paper 2',
        'sub_category': 'cs.CL',
        'abstract': 'Test abstract 2.',
        'published': datetime.now()
    }
    
    paper1_id = database.store_paper(paper1)
    paper2_id = database.store_paper(paper2)
    
    database.log_interaction(test_user_id, paper1_id, 'interested')
    database.log_interaction(test_user_id, paper2_id, 'not_interested')
    
    count = database.get_interaction_count(test_user_id)
    assert count == 2, f"Expected 2 interactions, got {count}"
    print("✓ get_interaction_count works correctly (with interactions)")
    
    # Test get_latest_papers
    latest = database.get_latest_papers(limit=10)
    assert len(latest) == 2, f"Expected 2 papers, got {len(latest)}"
    assert latest[0]['arxiv_id'] in ['2310.00001', '2310.00002']
    print("✓ get_latest_papers works correctly")


def test_select_diverse_papers():
    """Test the _select_diverse_papers function."""
    print("\n=== Test 2: Select Diverse Papers ===")
    
    # Create test papers with different categories
    test_papers = []
    categories = ['cs.AI', 'cs.CL', 'cs.CV', 'math.ST', 'physics.comp-ph']
    
    for i in range(30):
        category = categories[i % len(categories)]
        paper = {
            'id': i,
            'arxiv_id': f'2310.{i:05d}',
            'title': f'Test Paper {i}',
            'sub_category': category,
            'abstract': f'Test abstract {i}.',
            'published': datetime.now()
        }
        test_papers.append(paper)
    
    # Select 10 papers
    selected = manager._select_diverse_papers(test_papers, count=10)
    
    assert len(selected) == 10, f"Expected 10 papers, got {len(selected)}"
    
    # Check category diversity
    categories_in_selected = [p['sub_category'] for p in selected]
    unique_categories = set(categories_in_selected)
    assert len(unique_categories) >= 3, f"Expected at least 3 different categories, got {len(unique_categories)}"
    
    print(f"✓ Selected {len(selected)} papers with {len(unique_categories)} unique categories")
    
    # Test with fewer papers than requested
    few_papers = test_papers[:5]
    selected = manager._select_diverse_papers(few_papers, count=10)
    assert len(selected) == 5, f"Expected 5 papers (all available), got {len(selected)}"
    print("✓ Handles case with fewer papers than requested")


def test_get_matches_calibration_mode():
    """Test get_matches_for_user in calibration mode."""
    print("\n=== Test 3: Get Matches - Calibration Mode ===")
    setup_test_environment()
    
    # Add a user (defaults to calibration status)
    test_user_id = 12345
    database.add_user(test_user_id)
    
    # Add test papers
    categories = ['cs.AI', 'cs.CL', 'cs.CV', 'math.ST', 'physics.comp-ph']
    for i in range(50):
        category = categories[i % len(categories)]
        paper = {
            'arxiv_id': f'2310.{i:05d}',
            'title': f'Test Paper {i}',
            'sub_category': category,
            'abstract': f'Test abstract {i}.',
            'published': datetime.now()
        }
        database.store_paper(paper)
    
    # Get matches - should return 10 papers with category diversity
    matches = manager.get_matches_for_user(test_user_id)
    
    assert len(matches) <= 10, f"Expected at most 10 papers, got {len(matches)}"
    
    # Check category diversity
    categories_in_matches = [p.get('sub_category') for p in matches]
    unique_categories = set(categories_in_matches)
    print(f"✓ Got {len(matches)} matches with {len(unique_categories)} unique categories")
    
    # Verify user is still in calibration mode
    status = database.get_user_status(test_user_id)
    assert status == 'calibration', f"Expected 'calibration', got {status}"
    print("✓ User remains in calibration mode")


def test_get_matches_filters_seen_papers():
    """Test that get_matches_for_user filters out already-seen papers."""
    print("\n=== Test 4: Get Matches - Filters Seen Papers ===")
    setup_test_environment()
    
    # Add a user
    test_user_id = 12345
    database.add_user(test_user_id)
    
    # Add 20 test papers
    for i in range(20):
        paper = {
            'arxiv_id': f'2310.{i:05d}',
            'title': f'Test Paper {i}',
            'sub_category': 'cs.AI',
            'abstract': f'Test abstract {i}.',
            'published': datetime.now()
        }
        database.store_paper(paper)
    
    # Get initial matches
    matches1 = manager.get_matches_for_user(test_user_id)
    
    # Mark all returned papers as seen
    for paper in matches1:
        database.log_interaction(test_user_id, paper['id'], 'not_interested')
    
    # Get matches again - should return different papers
    matches2 = manager.get_matches_for_user(test_user_id)
    
    # Check that there's no overlap
    matches1_ids = {p['id'] for p in matches1}
    matches2_ids = {p['id'] for p in matches2}
    overlap = matches1_ids & matches2_ids
    
    assert len(overlap) == 0, f"Expected no overlap, but found {len(overlap)} common papers"
    print(f"✓ Filters out seen papers correctly")


def test_calibration_to_active_transition():
    """Test the automatic transition from calibration to active mode."""
    print("\n=== Test 5: Calibration to Active Transition ===")
    setup_test_environment()
    
    # Add a user
    test_user_id = 12345
    database.add_user(test_user_id)
    
    # Add 50 test papers
    for i in range(50):
        paper = {
            'arxiv_id': f'2310.{i:05d}',
            'title': f'Test Paper {i}',
            'sub_category': 'cs.AI',
            'abstract': f'Test abstract {i}.',
            'published': datetime.now()
        }
        database.store_paper(paper)
    
    # Verify user starts in calibration mode
    status = database.get_user_status(test_user_id)
    assert status == 'calibration', f"Expected 'calibration', got {status}"
    
    # Add exactly 21 interactions (just over the threshold of 20)
    for i in range(21):
        database.log_interaction(test_user_id, i + 1, 'interested')
    
    # Verify interaction count
    count = database.get_interaction_count(test_user_id)
    assert count == 21, f"Expected 21 interactions, got {count}"
    
    # Mock the brain module to avoid needing real embeddings
    with patch('manager.brain.RecommendationEngine') as MockEngine:
        mock_engine_instance = Mock()
        mock_engine_instance.update_user_profile.return_value = True
        mock_engine_instance.evaluate_paper.return_value = False  # Return False to simplify test
        MockEngine.return_value = mock_engine_instance
        
        # Call get_matches_for_user - should trigger transition
        matches = manager.get_matches_for_user(test_user_id)
        
        # Verify user is now in active mode
        status = database.get_user_status(test_user_id)
        assert status == 'active', f"Expected 'active', got {status}"
        print("✓ User transitioned from calibration to active mode")


def test_get_matches_active_mode():
    """Test get_matches_for_user in active mode."""
    print("\n=== Test 6: Get Matches - Active Mode ===")
    setup_test_environment()
    
    # Add a user and set to active mode
    test_user_id = 12345
    database.add_user(test_user_id)
    database.update_user_status(test_user_id, 'active')
    
    # Add some interested papers for the user profile
    for i in range(5):
        paper = {
            'arxiv_id': f'2310.{i:05d}',
            'title': f'Interested Paper {i}',
            'sub_category': 'cs.AI',
            'abstract': f'AI abstract {i}.',
            'published': datetime.now()
        }
        paper_id = database.store_paper(paper)
        database.log_interaction(test_user_id, paper_id, 'interested')
    
    # Add new unseen papers
    for i in range(5, 15):
        paper = {
            'arxiv_id': f'2310.{i:05d}',
            'title': f'New Paper {i}',
            'sub_category': 'cs.AI',
            'abstract': f'AI abstract {i}.',
            'published': datetime.now()
        }
        database.store_paper(paper)
    
    # Mock the brain module
    with patch('manager.brain.RecommendationEngine') as MockEngine:
        mock_engine_instance = Mock()
        mock_engine_instance.update_user_profile.return_value = True
        
        # Return True for first 3 papers, False for rest
        def mock_evaluate(user_id, paper):
            return paper['id'] in [6, 7, 8]
        
        mock_engine_instance.evaluate_paper.side_effect = mock_evaluate
        MockEngine.return_value = mock_engine_instance
        
        # Get matches - should use brain to filter
        matches = manager.get_matches_for_user(test_user_id)
        
        # Verify that update_user_profile was called
        mock_engine_instance.update_user_profile.assert_called_once_with(test_user_id)
        
        # Verify that evaluate_paper was called for unseen papers
        assert mock_engine_instance.evaluate_paper.call_count > 0
        
        # Verify matches contains only recommended papers
        match_ids = {p['id'] for p in matches}
        assert match_ids == {6, 7, 8}, f"Expected {{6, 7, 8}}, got {match_ids}"
        
        print(f"✓ Active mode uses brain to filter papers")
        print(f"  Returned {len(matches)} recommended papers")


def test_nonexistent_user():
    """Test behavior when user doesn't exist."""
    print("\n=== Test 7: Nonexistent User ===")
    setup_test_environment()
    
    # Try to get matches for a user that doesn't exist
    matches = manager.get_matches_for_user(99999)
    
    assert matches == [], f"Expected empty list, got {matches}"
    print("✓ Returns empty list for nonexistent user")


if __name__ == "__main__":
    """Run all tests."""
    print("=" * 80)
    print("Running Manager Module Tests")
    print("=" * 80)
    
    try:
        test_database_helper_functions()
        test_select_diverse_papers()
        test_get_matches_calibration_mode()
        test_get_matches_filters_seen_papers()
        test_calibration_to_active_transition()
        test_get_matches_active_mode()
        test_nonexistent_user()
        
        print("\n" + "=" * 80)
        print("All Tests Passed!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
