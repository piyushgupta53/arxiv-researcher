"""
Custom toolkit for interacting with arXiv API and parsing PDFs.
"""

from typing import List, Optional
import arxiv
from agno.agent import Agent
from pypdf import PdfReader
import requests


class ArxivTools:
    """Tools for searching arXiv and reading papers."""

    def __init__(self, search_arxiv: bool = True, read_arxiv_papers: bool = True):
        """Initialize with selected capabilities."""
        self.tools = []
        if search_arxiv:
            self.tools.append(self.search_arxiv_and_return_articles)
        if read_arxiv_papers:
            self.tools.append(self.read_arxiv_papers)

    def search_arxiv_and_return_articles(
        self, query: str, num_articles: int = 40
    ) -> dict:
        """Search arXiv for articles matching the query.

        Args:
            query: The search query
            num_articles: Maximum number of articles to return (default: 40)

        Returns:
            Dictionary containing list of papers with their metadata
        """
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=num_articles, sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for result in client.results(search):
            papers.append(
                {
                    "id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "published": (
                        result.published.isoformat() if result.published else None
                    ),
                }
            )
        return {"papers": papers}

    def read_arxiv_papers(
        self, id_list: List[str], pages_to_read: int = 6
    ) -> List[dict]:
        """Download and extract text from the first N pages of arXiv papers.

        Args:
            id_list: List of arXiv IDs
            pages_to_read: Number of pages to extract from each paper

        Returns:
            List of papers with their extracted text content
        """
        papers = []
        for arxiv_id in id_list:
            # Get paper metadata
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search))

            # Download PDF
            response = requests.get(paper.pdf_url)
            if response.status_code != 200:
                continue

            # Extract text from pages
            pages = []
            try:
                reader = PdfReader(response.raw)
                for i in range(min(pages_to_read, len(reader.pages))):
                    text = reader.pages[i].extract_text()
                    if text.strip():
                        pages.append({"id": arxiv_id, "page": i + 1, "text": text})
            except Exception as e:
                print(f"Error processing {arxiv_id}: {e}")
                continue

            papers.append({"id": arxiv_id, "title": paper.title, "pages": pages})

        return papers
