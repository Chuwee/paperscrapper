"""
Brain module for recommendation intelligence using Embeddings and LLM.
"""
import os
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv

import database

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.3
# Use absolute path in the current working directory
CHROMA_DB_PATH = os.path.abspath("./chroma_db")


class RecommendationEngine:
    """
    Recommendation engine that uses embeddings for fast filtering
    and LLM for final decision-making.
    """
    
    def __init__(self, mock_llm: bool = False, use_ephemeral: bool = False):
        """
        Initialize the recommendation engine.
        
        Args:
            mock_llm: If True, use mock LLM responses for testing
            use_ephemeral: If True, use ephemeral ChromaDB client (in-memory)
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB client
        if use_ephemeral:
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection for user profiles
        self.user_profiles_collection = self.chroma_client.get_or_create_collection(
            name="user_profiles",
            metadata={"description": "User interest vectors based on liked papers"}
        )
        
        # Mock LLM setting
        self.mock_llm = mock_llm
        
        # Initialize OpenAI client if not mocking
        if not self.mock_llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
            else:
                self.openai_client = None
        else:
            self.openai_client = None
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text string.
        
        Args:
            text: Input text to embed
        
        Returns:
            Numpy array of embedding vector
        """
        embedding = self.embedding_model.encode(text)
        return embedding
    
    def update_user_profile(self, user_id: int) -> bool:
        """
        Update user profile by fetching all interested papers and creating a user vector.
        
        Args:
            user_id: The telegram_id of the user
        
        Returns:
            True if profile was updated successfully, False otherwise
        """
        # Fetch all interested papers for this user
        interested_papers = database.get_interested_papers(user_id)
        
        if not interested_papers:
            # No interested papers yet, cannot create profile
            return False
        
        # Generate embeddings for all interested paper abstracts
        embeddings = []
        for paper in interested_papers:
            abstract = paper.get('abstract', '')
            if abstract:
                embedding = self._generate_embedding(abstract)
                embeddings.append(embedding)
        
        if not embeddings:
            return False
        
        # Average embeddings to create user vector
        user_vector = np.mean(embeddings, axis=0)
        
        # Store user vector in ChromaDB
        user_id_str = str(user_id)
        
        # Check if user profile already exists
        try:
            self.user_profiles_collection.delete(ids=[user_id_str])
        except Exception:
            pass  # User profile doesn't exist yet
        
        # Add the new user profile
        self.user_profiles_collection.add(
            embeddings=[user_vector.tolist()],
            ids=[user_id_str],
            metadatas=[{"user_id": user_id, "num_papers": len(interested_papers)}]
        )
        
        return True
    
    def _get_user_vector(self, user_id: int) -> Optional[np.ndarray]:
        """
        Retrieve user vector from ChromaDB.
        
        Args:
            user_id: The telegram_id of the user
        
        Returns:
            User vector as numpy array, or None if not found
        """
        user_id_str = str(user_id)
        
        try:
            results = self.user_profiles_collection.get(
                ids=[user_id_str],
                include=["embeddings"]
            )
            if results['embeddings'] is not None and len(results['embeddings']) > 0:
                embedding = results['embeddings'][0]
                # ChromaDB returns embeddings as numpy arrays or lists
                if isinstance(embedding, list):
                    return np.array(embedding)
                return embedding
        except Exception as e:
            # For debugging
            print(f"Error retrieving user vector: {e}")
        
        return None
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return float(similarity)
    
    def _call_llm(self, user_papers: List[Dict], current_paper: Dict) -> bool:
        """
        Call LLM to make final recommendation decision.
        
        Args:
            user_papers: List of user's interested papers (top 3)
            current_paper: The paper to evaluate
        
        Returns:
            True if LLM recommends the paper, False otherwise
        """
        if self.mock_llm:
            # Mock response for testing - return True if similarity is high enough
            # This is a simple heuristic for testing
            return True
        
        # Prepare the prompt
        user_titles = "\n".join([f"- {p['title']}" for p in user_papers[:3]])
        current_abstract = current_paper.get('abstract', '')
        
        system_prompt = "You are a research assistant. Based on the user's history, decide if this new paper is relevant. Reply ONLY with boolean TRUE or FALSE."
        
        user_prompt = f"""User's previously liked papers:
{user_titles}

New paper abstract:
{current_abstract}

Is this new paper relevant to the user's interests?"""
        
        try:
            if not self.openai_client:
                # No API key configured, default to False
                print("OpenAI client not initialized (no API key)")
                return False
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return "TRUE" in answer
            
        except Exception as e:
            # If LLM call fails, default to False (conservative)
            print(f"LLM call failed: {e}")
            return False
    
    def evaluate_paper(self, user_id: int, paper: Dict) -> bool:
        """
        Evaluate a paper using 2-step pipeline: vector similarity + LLM.
        
        Args:
            user_id: The telegram_id of the user
            paper: Dictionary containing paper data (must have 'abstract' key)
        
        Returns:
            True if paper is recommended, False otherwise
        """
        # Get user vector
        user_vector = self._get_user_vector(user_id)
        
        if user_vector is None:
            # User has no profile yet, cannot evaluate
            return False
        
        # Step 1: Fast vector similarity filter
        abstract = paper.get('abstract', '')
        if not abstract:
            return False
        
        paper_embedding = self._generate_embedding(abstract)
        similarity = self._calculate_similarity(user_vector, paper_embedding)
        
        if similarity < SIMILARITY_THRESHOLD:
            # Paper doesn't pass similarity threshold
            return False
        
        # Step 2: LLM judge
        # Get user's interested papers for context
        interested_papers = database.get_interested_papers(user_id)
        
        if not interested_papers:
            # Shouldn't happen if user_vector exists, but handle gracefully
            return False
        
        # Call LLM with top 3 papers
        return self._call_llm(interested_papers[:3], paper)


def get_paper_score(user_id: int, paper_text: str, mock_llm: bool = False) -> bool:
    """
    Convenience function to get paper recommendation score.
    
    Args:
        user_id: The telegram_id of the user
        paper_text: The paper abstract or full text
        mock_llm: If True, use mock LLM responses for testing
    
    Returns:
        True if paper is recommended, False otherwise
    """
    engine = RecommendationEngine(mock_llm=mock_llm)
    paper = {'abstract': paper_text}
    return engine.evaluate_paper(user_id, paper)


if __name__ == "__main__":
    """
    Test script to verify brain functionality.
    """
    import os
    from datetime import datetime
    
    # Clean up test databases
    if os.path.exists(database.DB_PATH):
        os.remove(database.DB_PATH)
        print(f"Removed existing {database.DB_PATH}")
    
    import shutil
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Removed existing {CHROMA_DB_PATH}")
    
    # Initialize database
    print("\n=== Test 1: Initialize Database ===")
    database.init_db()
    database.add_user(12345)
    print("✓ Database initialized and user added")
    
    # Add some test papers
    print("\n=== Test 2: Add Test Papers ===")
    paper1 = {
        'arxiv_id': '2310.00001',
        'title': 'Deep Learning for Natural Language Processing',
        'sub_category': 'cs.CL',
        'abstract': 'This paper presents a comprehensive study of deep learning techniques for natural language processing tasks, including sentiment analysis and machine translation.',
        'published': datetime.now()
    }
    paper2 = {
        'arxiv_id': '2310.00002',
        'title': 'Transformer Models in NLP',
        'sub_category': 'cs.CL',
        'abstract': 'We explore the application of transformer architectures in various NLP tasks, demonstrating state-of-the-art performance on benchmark datasets.',
        'published': datetime.now()
    }
    paper3 = {
        'arxiv_id': '2310.00003',
        'title': 'Quantum Computing Algorithms',
        'sub_category': 'quant-ph',
        'abstract': 'This work presents novel quantum computing algorithms for solving optimization problems in polynomial time.',
        'published': datetime.now()
    }
    
    paper1_id = database.store_paper(paper1)
    paper2_id = database.store_paper(paper2)
    paper3_id = database.store_paper(paper3)
    print(f"✓ Added 3 test papers (IDs: {paper1_id}, {paper2_id}, {paper3_id})")
    
    # Mark papers 1 and 2 as interested
    print("\n=== Test 3: Log User Interests ===")
    database.log_interaction(12345, paper1_id, 'interested')
    database.log_interaction(12345, paper2_id, 'interested')
    print("✓ User 12345 interested in papers 1 and 2 (NLP papers)")
    
    # Test 4: Create recommendation engine
    print("\n=== Test 4: Initialize Recommendation Engine ===")
    engine = RecommendationEngine(mock_llm=True)
    print("✓ Recommendation engine initialized with mock LLM")
    
    # Test 5: Generate embedding
    print("\n=== Test 5: Test Embedding Generation ===")
    test_text = "This is a test sentence for embedding."
    embedding = engine._generate_embedding(test_text)
    print(f"✓ Generated embedding of shape: {embedding.shape}")
    print(f"  Embedding dimension: {len(embedding)}")
    
    # Test 6: Update user profile
    print("\n=== Test 6: Update User Profile ===")
    success = engine.update_user_profile(12345)
    print(f"✓ User profile updated: {success}")
    
    # Verify user vector is stored
    user_vector = engine._get_user_vector(12345)
    print(f"  Retrieved user vector shape: {user_vector.shape if user_vector is not None else 'None'}")
    
    # Test 7: Evaluate similar paper (should recommend)
    print("\n=== Test 7: Evaluate Similar Paper (NLP) ===")
    similar_paper = {
        'arxiv_id': '2310.00004',
        'title': 'Attention Mechanisms in Language Models',
        'abstract': 'We study attention mechanisms in large language models and their impact on understanding natural language semantics.',
    }
    result = engine.evaluate_paper(12345, similar_paper)
    print(f"  Recommendation: {result} (expected: True)")
    
    # Test 8: Evaluate dissimilar paper (should not recommend)
    print("\n=== Test 8: Evaluate Dissimilar Paper (Quantum) ===")
    dissimilar_paper = {
        'arxiv_id': '2310.00005',
        'title': 'Quantum Entanglement Properties',
        'abstract': 'This paper explores quantum entanglement phenomena in multi-qubit systems and their applications in quantum cryptography.',
    }
    result = engine.evaluate_paper(12345, dissimilar_paper)
    print(f"  Recommendation: {result} (expected: False)")
    
    # Test 9: Test get_paper_score function
    print("\n=== Test 9: Test get_paper_score Function ===")
    score = get_paper_score(
        12345,
        "Novel approaches to sequence-to-sequence learning using neural networks for machine translation tasks.",
        mock_llm=True
    )
    print(f"  Paper score: {score}")
    
    print("\n=== All Tests Completed! ===")
