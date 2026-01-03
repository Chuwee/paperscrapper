"""
Unit tests for brain module.
"""
import os
import sys
import shutil
import sqlite3
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import database
from brain import RecommendationEngine, get_paper_score, CHROMA_DB_PATH


def setup_test_environment():
    """Set up a clean test environment."""
    # Clean up test databases
    if os.path.exists(database.DB_PATH):
        os.remove(database.DB_PATH)
    
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Initialize database
    database.init_db()


def test_database_helper():
    """Test the get_interested_papers database helper function."""
    print("\n=== Test 1: Database Helper Function ===")
    setup_test_environment()
    
    # Add a user
    database.add_user(12345)
    
    # Add some papers
    paper1 = {
        'arxiv_id': '2310.00001',
        'title': 'Deep Learning for NLP',
        'sub_category': 'cs.CL',
        'abstract': 'This paper presents deep learning for NLP.',
        'published': datetime.now()
    }
    paper2 = {
        'arxiv_id': '2310.00002',
        'title': 'Quantum Computing',
        'sub_category': 'quant-ph',
        'abstract': 'This paper presents quantum computing algorithms.',
        'published': datetime.now()
    }
    
    paper1_id = database.store_paper(paper1)
    paper2_id = database.store_paper(paper2)
    
    # Mark paper 1 as interested, paper 2 as not interested
    database.log_interaction(12345, paper1_id, 'interested')
    database.log_interaction(12345, paper2_id, 'not_interested')
    
    # Get interested papers
    interested = database.get_interested_papers(12345)
    
    assert len(interested) == 1, f"Expected 1 interested paper, got {len(interested)}"
    assert interested[0]['arxiv_id'] == '2310.00001', "Wrong paper returned"
    print("✓ Database helper function works correctly")


def test_embedding_generation():
    """Test embedding generation with mocked SentenceTransformer."""
    print("\n=== Test 2: Embedding Generation (Mocked) ===")
    setup_test_environment()
    
    # Mock SentenceTransformer
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        MockSentenceTransformer.return_value = mock_model
        
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        
        # Test embedding generation
        embedding = engine._generate_embedding("test text")
        
        assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
        assert len(embedding) == 5, f"Expected 5 dimensions, got {len(embedding)}"
        print(f"✓ Generated embedding with shape: {embedding.shape}")


def test_user_profile_update():
    """Test user profile update with mocked embeddings."""
    print("\n=== Test 3: User Profile Update (Mocked) ===")
    setup_test_environment()
    
    # Add user and papers
    database.add_user(12345)
    
    paper1 = {
        'arxiv_id': '2310.00001',
        'title': 'Deep Learning for NLP',
        'sub_category': 'cs.CL',
        'abstract': 'This paper presents deep learning for NLP.',
        'published': datetime.now()
    }
    paper2 = {
        'arxiv_id': '2310.00002',
        'title': 'Transformer Models',
        'sub_category': 'cs.CL',
        'abstract': 'This paper explores transformer architectures.',
        'published': datetime.now()
    }
    
    paper1_id = database.store_paper(paper1)
    paper2_id = database.store_paper(paper2)
    
    database.log_interaction(12345, paper1_id, 'interested')
    database.log_interaction(12345, paper2_id, 'interested')
    
    # Mock SentenceTransformer
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # First paper embedding
            np.array([0.0, 1.0, 0.0]),  # Second paper embedding
        ]
        MockSentenceTransformer.return_value = mock_model
        
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        
        # Update user profile
        success = engine.update_user_profile(12345)
        
        assert success == True, "Profile update should succeed"
        
        # Verify user vector is stored
        user_vector = engine._get_user_vector(12345)
        assert user_vector is not None, "User vector should be stored"
        assert len(user_vector) == 3, f"Expected 3 dimensions, got {len(user_vector)}"
        
        # The user vector should be the average of the two embeddings
        expected_vector = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(user_vector, expected_vector)
        
        print(f"✓ User profile updated successfully")
        print(f"  User vector: {user_vector}")


def test_similarity_calculation():
    """Test cosine similarity calculation."""
    print("\n=== Test 4: Similarity Calculation ===")
    setup_test_environment()
    
    # Mock SentenceTransformer to avoid network calls
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        MockSentenceTransformer.return_value = mock_model
        
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        
        # Test similarity between identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = engine._calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001, f"Identical vectors should have similarity 1.0, got {similarity}"
        
        # Test similarity between orthogonal vectors
        vec3 = np.array([1.0, 0.0, 0.0])
        vec4 = np.array([0.0, 1.0, 0.0])
        similarity = engine._calculate_similarity(vec3, vec4)
        assert abs(similarity - 0.0) < 0.001, f"Orthogonal vectors should have similarity 0.0, got {similarity}"
        
        # Test similarity between opposite vectors
        vec5 = np.array([1.0, 0.0, 0.0])
        vec6 = np.array([-1.0, 0.0, 0.0])
        similarity = engine._calculate_similarity(vec5, vec6)
        assert abs(similarity - (-1.0)) < 0.001, f"Opposite vectors should have similarity -1.0, got {similarity}"
        
        print("✓ Similarity calculation works correctly")


def test_evaluate_paper():
    """Test paper evaluation with mocked components."""
    print("\n=== Test 5: Paper Evaluation (Mocked) ===")
    setup_test_environment()
    
    # Add user and interested papers
    database.add_user(12345)
    
    paper1 = {
        'arxiv_id': '2310.00001',
        'title': 'Deep Learning for NLP',
        'sub_category': 'cs.CL',
        'abstract': 'This paper presents deep learning for natural language processing.',
        'published': datetime.now()
    }
    
    paper1_id = database.store_paper(paper1)
    database.log_interaction(12345, paper1_id, 'interested')
    
    # Mock SentenceTransformer
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        
        # First call: update_user_profile (encoding interested paper)
        # Second call: evaluate_paper (encoding new paper)
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # Interested paper embedding
            np.array([0.9, 0.1, 0.0]),  # Similar new paper embedding (high similarity)
        ]
        MockSentenceTransformer.return_value = mock_model
        
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        
        # Update user profile
        engine.update_user_profile(12345)
        
        # Evaluate a similar paper
        new_paper = {
            'arxiv_id': '2310.00002',
            'title': 'Advanced NLP Techniques',
            'abstract': 'This paper explores advanced techniques in natural language processing.',
        }
        
        result = engine.evaluate_paper(12345, new_paper)
        
        # Should pass both similarity and mock LLM checks
        assert result == True, f"Similar paper should be recommended, got {result}"
        print("✓ Paper evaluation works correctly (high similarity)")
    
    # Test with low similarity
    setup_test_environment()
    database.add_user(12345)
    paper1_id_new = database.store_paper(paper1)
    database.log_interaction(12345, paper1_id_new, 'interested')
    
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # Interested paper embedding
            np.array([0.0, 1.0, 0.0]),  # Dissimilar new paper embedding (low similarity)
        ]
        MockSentenceTransformer.return_value = mock_model
        
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        engine.update_user_profile(12345)
        
        dissimilar_paper = {
            'arxiv_id': '2310.00003',
            'title': 'Quantum Computing Algorithms',
            'abstract': 'This paper presents quantum computing algorithms.',
        }
        
        result = engine.evaluate_paper(12345, dissimilar_paper)
        
        # Should fail similarity check
        assert result == False, f"Dissimilar paper should not be recommended, got {result}"
        print("✓ Paper evaluation works correctly (low similarity)")


def test_get_paper_score():
    """Test the convenience function get_paper_score."""
    print("\n=== Test 6: get_paper_score Function (Mocked) ===")
    setup_test_environment()
    
    # Add user and interested paper
    database.add_user(12345)
    
    paper1 = {
        'arxiv_id': '2310.00001',
        'title': 'Deep Learning for NLP',
        'sub_category': 'cs.CL',
        'abstract': 'This paper presents deep learning for natural language processing.',
        'published': datetime.now()
    }
    
    paper1_id = database.store_paper(paper1)
    database.log_interaction(12345, paper1_id, 'interested')
    
    # Mock SentenceTransformer
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # Interested paper
            np.array([0.95, 0.05, 0.0]),  # New paper (similar)
        ]
        MockSentenceTransformer.return_value = mock_model
        
        # Need to update profile first
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        engine.update_user_profile(12345)
        
        # Now test get_paper_score
        mock_model.encode.return_value = np.array([0.95, 0.05, 0.0])
        score = get_paper_score(12345, "Advanced NLP techniques using transformers.", mock_llm=True)
        
        assert isinstance(score, bool), "Score should be a boolean"
        print(f"✓ get_paper_score function works correctly: {score}")


def test_no_user_profile():
    """Test behavior when user has no profile yet."""
    print("\n=== Test 7: No User Profile ===")
    setup_test_environment()
    
    database.add_user(12345)
    
    with patch('brain.SentenceTransformer') as MockSentenceTransformer:
        mock_model = Mock()
        # Set a default return value for encode in case it's called
        mock_model.encode.return_value = np.array([1.0, 0.0, 0.0])
        MockSentenceTransformer.return_value = mock_model
        
        engine = RecommendationEngine(mock_llm=True, use_ephemeral=True)
        
        # Try to update profile with no interested papers
        success = engine.update_user_profile(12345)
        assert success == False, "Should fail when no interested papers"
        
        # Try to evaluate a paper with no profile
        paper = {'abstract': 'Test abstract'}
        result = engine.evaluate_paper(12345, paper)
        assert result == False, "Should return False when no user profile"
        
        print("✓ Handles missing user profile correctly")


if __name__ == "__main__":
    """Run all tests."""
    print("=" * 80)
    print("Running Brain Module Tests")
    print("=" * 80)
    
    try:
        test_database_helper()
        test_embedding_generation()
        test_user_profile_update()
        test_similarity_calculation()
        test_evaluate_paper()
        test_get_paper_score()
        test_no_user_profile()
        
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
