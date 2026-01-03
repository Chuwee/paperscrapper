"""
Harvester module for fetching papers from arXiv.
"""
import arxiv
import re


def fetch_papers(categories: list[str], max_results: int = 20) -> list[dict]:
    """
    Fetch papers from arXiv for specified categories.
    
    Args:
        categories: List of category strings (e.g., ['cs.CL', 'cs.LG', 'stat.ML'])
        max_results: Maximum number of results to return (default: 20)
    
    Returns:
        List of dictionaries containing paper metadata with keys:
        - arxiv_id: Unique ID (e.g., '2310.12345')
        - title: Paper title
        - abstract: Paper abstract (cleaned)
        - authors: List of author names
        - url: Direct link to the abstract page
        - published: Datetime object
        - categories: List of categories the paper belongs to
    """
    # Build search query for categories
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    # Create search client
    search = arxiv.Search(
        query=category_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Fetch and structure results
    papers = []
    for result in search.results():
        # Extract arXiv ID and remove version suffix (e.g., '2310.12345v1' -> '2310.12345')
        # Version suffix is 'v' followed by digits at the end
        short_id = result.get_short_id()
        arxiv_id = re.sub(r'v\d+$', '', short_id)
        
        paper = {
            'arxiv_id': arxiv_id,
            'title': result.title,
            'abstract': result.summary.replace('\n', ' ').strip(),  # Clean up newlines
            'authors': [author.name for author in result.authors],
            'url': result.entry_id,
            'published': result.published,
            'categories': result.categories
        }
        papers.append(paper)
    
    return papers


if __name__ == "__main__":
    # Verification: Test connectivity by fetching cs.AI papers
    print("Fetching papers from arXiv (cs.AI category)...")
    papers = fetch_papers(['cs.AI'], max_results=5)
    
    print(f"\nSuccessfully fetched {len(papers)} papers:")
    print("=" * 80)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
    print("=" * 80)
