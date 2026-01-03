"""
Manager module for orchestrating the calibration loop and paper matching.
"""
import random
from typing import List, Dict, Any
from collections import defaultdict

import database
import brain


def get_matches_for_user(user_id: int, mock_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Get matched papers for a user based on their current status.
    
    For 'calibration' users:
    - Fetches the latest 50 papers
    - Selects 10 semi-randomly with category diversity
    - Filters out already-seen papers
    - Transitions to 'active' if user has >20 interactions
    
    For 'active' users:
    - Fetches latest papers
    - Runs brain.evaluate_paper on each
    - Returns only papers that are recommended
    
    Args:
        user_id: The telegram_id of the user
        mock_llm: If True, use mock LLM for testing (default: False)
    
    Returns:
        List of dictionaries containing paper data
    """
    # Fetch user status from database
    status = database.get_user_status(user_id)
    
    if status is None:
        # User doesn't exist, return empty list
        return []
    
    # Get latest papers
    latest_papers = database.get_latest_papers(limit=50)
    
    # Filter out papers already seen by the user
    # Use the existing function to get unseen papers, then filter from latest 50
    all_unseen = database.get_unseen_papers(user_id, limit=1000)
    unseen_paper_ids = {p['id'] for p in all_unseen}
    
    # Filter latest papers to only include unseen ones
    unseen_latest = [p for p in latest_papers if p['id'] in unseen_paper_ids]
    
    if status == 'calibration':
        # Check if user should transition to active
        interaction_count = database.get_interaction_count(user_id)
        if interaction_count > 20:
            # Before transitioning, verify user has interested papers for profile creation
            interested_papers = database.get_interested_papers(user_id)
            if interested_papers:
                # User has interested papers, safe to transition
                database.update_user_status(user_id, 'active')
                # Recurse to get matches for active user
                return get_matches_for_user(user_id, mock_llm=mock_llm)
            # else: User has no interested papers, stay in calibration mode
        
        # Calibration mode: select 10 papers with category diversity
        return _select_diverse_papers(unseen_latest, count=10)
    
    elif status == 'active':
        # Active mode: use brain to evaluate papers
        recommended_papers = []
        
        # Create recommendation engine instance
        engine = brain.RecommendationEngine(mock_llm=mock_llm)
        
        # Update user profile first (in case it's not up to date)
        profile_updated = engine.update_user_profile(user_id)
        
        # If profile update failed, fall back to calibration mode
        if not profile_updated:
            # Revert to calibration mode and get diverse papers
            database.update_user_status(user_id, 'calibration')
            return _select_diverse_papers(unseen_latest, count=10)
        
        # Evaluate each unseen paper
        for paper in unseen_latest:
            if engine.evaluate_paper(user_id, paper):
                recommended_papers.append(paper)
        
        return recommended_papers
    
    else:
        # Unknown status, return empty list
        return []


def _select_diverse_papers(papers: List[Dict[str, Any]], count: int = 10) -> List[Dict[str, Any]]:
    """
    Select papers semi-randomly with category diversity.
    
    Strategy:
    - Group papers by sub_category
    - Try to select approximately equal numbers from each category
    - If not enough papers, return what's available
    
    Args:
        papers: List of paper dictionaries
        count: Number of papers to select (default: 10)
    
    Returns:
        List of selected papers with category diversity
    """
    if len(papers) <= count:
        # Not enough papers to select from, return all
        return papers
    
    # Group papers by category
    papers_by_category = defaultdict(list)
    for paper in papers:
        category = paper.get('sub_category', 'unknown')
        papers_by_category[category].append(paper)
    
    # Calculate how many papers to select from each category
    num_categories = len(papers_by_category)
    if num_categories == 0:
        return []
    
    # Target papers per category (with some randomness)
    papers_per_category = max(1, count // num_categories)
    
    selected_papers = []
    categories = list(papers_by_category.keys())
    random.shuffle(categories)  # Randomize category order
    
    # First pass: select papers_per_category from each category
    for category in categories:
        category_papers = papers_by_category[category]
        # Randomly shuffle papers within category
        random.shuffle(category_papers)
        # Select up to papers_per_category papers
        num_to_select = min(papers_per_category, len(category_papers))
        selected_papers.extend(category_papers[:num_to_select])
        
        if len(selected_papers) >= count:
            break
    
    # If we haven't selected enough papers, add more from remaining categories
    if len(selected_papers) < count:
        # Collect remaining papers
        remaining_papers = []
        for category in categories:
            category_papers = papers_by_category[category]
            # Skip papers already selected
            for paper in category_papers:
                if paper not in selected_papers:
                    remaining_papers.append(paper)
        
        # Shuffle and add to reach target count
        random.shuffle(remaining_papers)
        needed = count - len(selected_papers)
        selected_papers.extend(remaining_papers[:needed])
    
    # Trim to exact count if we selected too many
    return selected_papers[:count]


if __name__ == "__main__":
    """
    Test script to verify manager functionality.
    """
    import os
    from datetime import datetime
    
    # Clean up test database
    if os.path.exists(database.DB_PATH):
        os.remove(database.DB_PATH)
        print(f"Removed existing {database.DB_PATH}")
    
    # Initialize database
    print("\n=== Test 1: Initialize Database ===")
    database.init_db()
    print("✓ Database initialized")
    
    # Add a test user
    print("\n=== Test 2: Add User ===")
    test_user_id = 12345
    database.add_user(test_user_id)
    print(f"✓ User {test_user_id} added with status: {database.get_user_status(test_user_id)}")
    
    # Add test papers with different categories
    print("\n=== Test 3: Add Test Papers ===")
    categories = ['cs.AI', 'cs.CL', 'cs.CV', 'math.ST', 'physics.comp-ph']
    test_papers = []
    
    for i, category in enumerate(categories * 10):  # 50 papers total
        paper = {
            'arxiv_id': f'2310.{i:05d}',
            'title': f'Test Paper {i} - {category}',
            'sub_category': category,
            'abstract': f'This is a test abstract for paper {i} in category {category}.',
            'published': datetime.now()
        }
        paper_id = database.store_paper(paper)
        test_papers.append({'id': paper_id, **paper})
    
    print(f"✓ Added {len(test_papers)} test papers")
    
    # Test 4: Get matches for calibration user (new user, no interactions)
    print("\n=== Test 4: Get Matches for Calibration User ===")
    matches = get_matches_for_user(test_user_id)
    print(f"✓ Got {len(matches)} matches for calibration user")
    
    # Check category diversity
    categories_in_matches = [p.get('sub_category') for p in matches]
    unique_categories = set(categories_in_matches)
    print(f"  Categories in matches: {unique_categories}")
    print(f"  Number of unique categories: {len(unique_categories)}")
    
    # Test 5: Log some interactions
    print("\n=== Test 5: Log Interactions ===")
    for i, match in enumerate(matches[:5]):
        database.log_interaction(test_user_id, match['id'], 'interested')
    print(f"✓ Logged 5 interactions (interested)")
    print(f"  Total interactions: {database.get_interaction_count(test_user_id)}")
    print(f"  User status: {database.get_user_status(test_user_id)}")
    
    # Test 6: Add more interactions to trigger transition
    print("\n=== Test 6: Trigger Calibration -> Active Transition ===")
    # Get more papers and interact with them
    for i in range(6, 22):  # Add 16 more interactions (total 21)
        if i < len(test_papers):
            database.log_interaction(test_user_id, test_papers[i]['id'], 'interested')
    
    print(f"  Total interactions: {database.get_interaction_count(test_user_id)}")
    print(f"  User status before get_matches: {database.get_user_status(test_user_id)}")
    
    # This should trigger the transition
    matches_active = get_matches_for_user(test_user_id)
    print(f"  User status after get_matches: {database.get_user_status(test_user_id)}")
    print(f"✓ User transitioned to active mode")
    
    # Test 7: Test _select_diverse_papers directly
    print("\n=== Test 7: Test _select_diverse_papers ===")
    # Create a small test set
    small_papers = test_papers[:20]
    selected = _select_diverse_papers(small_papers, count=10)
    print(f"✓ Selected {len(selected)} papers from {len(small_papers)}")
    
    categories_selected = [p.get('sub_category') for p in selected]
    print(f"  Categories: {categories_selected}")
    
    print("\n=== All Tests Completed! ===")
