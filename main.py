"""
End‑to‑end arXiv literature‑review workflow built with Agno Workflow.
Pipeline outline:
1. Search arXiv (up to 40 hits).
2. Embed title + abstract, rank semantically and keep the dynamic top‑k (≤ 12).
3. Download & parse the first N pages of each paper (default 6).
4. Build a FAISS index over page chunks for retrieval‑augmented generation.
5. Ask DeepSeek (PaperSynth) to write a cited markdown report.
6. Persist intermediate artefacts in SQLite‑backed session_state caches.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Iterator, List, Optional, Tuple
import numpy as np
import faiss
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from arxiv_tools import ArxivTools

# Initialize Rich console for better output
console = Console()

################################################################################
# Typed payloads
################################################################################


class PaperMeta(BaseModel):
    id: str
    title: str
    summary: str
    pdf_url: str
    published: Optional[str]


class SearchOutput(BaseModel):
    papers: List[PaperMeta]


class PageChunk(BaseModel):
    id: str
    page: int
    text: str


class ParsedPaper(BaseModel):
    id: str
    title: str
    pages: List[PageChunk]


class ReadOutput(BaseModel):
    papers: List[ParsedPaper]


################################################################################
# Embedding & chunking helpers
################################################################################

_EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Loads (or memoises) the sentence‑transformers model used across stages."""
    global _embedder
    if _embedder is None:
        console.print(
            f"[bold blue]Loading embedding model: {_EMB_MODEL_NAME}[/bold blue]"
        )
        _embedder = SentenceTransformer(_EMB_MODEL_NAME)
        console.print("[bold green]✓ Embedding model loaded successfully[/bold green]")
    return _embedder


CHUNK_SIZE = 1_000  # characters
CHUNK_OVERLAP = 200


def chunk_text(text: str) -> List[str]:
    """Naïve character‑based chunker (good enough for short PDF pages)."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP if end < len(text) else end
    return chunks


################################################################################
# Agents
################################################################################

# Searcher
searcher = Agent(
    model=DeepSeek(id="deepseek-chat"),
    tools=[ArxivTools(search_arxiv=True, read_arxiv_papers=False)],
    description="arXiv‑Scout: finds candidate papers for a technical query.",
    instructions=dedent(
        """
        1. Call search_arxiv_and_return_articles with num_articles = 40.
        2. Return the JSON list under the key papers.
        """
    ),
    response_model=SearchOutput,
)

# Reader
reader = Agent(
    model=DeepSeek(id="deepseek-chat"),
    tools=[ArxivTools(search_arxiv=False, read_arxiv_papers=True)],
    description="PDF‑Extractor: downloads papers and extracts the first N pages.",
    instructions=dedent(
        """
        Input: list of arXiv IDs.
        Call read_arxiv_papers(id_list, pages_to_read=6) and return the parsed
        papers as structured data under the key papers.
        """
    ),
    response_model=ReadOutput,
    structured_outputs=True,
)

# Writer
writer = Agent(
    model=DeepSeek(id="deepseek-chat"),
    description="PaperSynth: writes a comprehensive, cited literature review that adapts to the research question and paper content.",
    instructions=dedent(
        """
        You will receive a JSON object containing:
        1. The user's research question
        2. Selected page excerpts from arXiv papers
        3. Paper metadata (titles, authors, etc.)

        Your task is to write a comprehensive literature review that:
        1. Directly addresses the user's research question
        2. Synthesizes findings across papers
        3. Identifies patterns, contradictions, and gaps
        4. Provides actionable insights

        Structure your report as follows:

        # Executive Summary (200-300 words)
        - Context and motivation for the research
        - Key findings and trends
        - Main challenges and opportunities
        - Overall implications

        # Methodology Analysis
        - Common approaches and techniques
        - Key innovations and breakthroughs
        - Technical requirements and constraints
        - Implementation considerations

        # Comparative Analysis
        Create a detailed comparison table focusing on metrics relevant to the research question:
        | Approach | Strengths | Limitations | Performance Metrics | Implementation Complexity |
        |----------|-----------|-------------|-------------------|-------------------------|
        [Fill with data from papers]

        # Critical Discussion
        - Emerging trends and patterns
        - Conflicting findings or approaches
        - Technical challenges and solutions
        - Scalability and practical considerations

        # Research Gaps and Future Directions
        - Identified limitations in current approaches
        - Promising research directions
        - Open technical challenges
        - Potential impact areas

        # References
        For each paper, provide:
        - Full citation
        - Key contribution to the field
        - Relevance to the research question

        Guidelines:
        1. Ground all statements in the provided papers using [id:page] citations
        2. Highlight both consensus and disagreements among papers
        3. Focus on practical implications and implementation considerations
        4. Adapt the analysis depth based on paper count and content
        5. If papers cover different aspects, organize by themes rather than chronologically
        6. Include quantitative comparisons where available
        7. Note any assumptions or limitations in the analysis
        """
    ),
    markdown=True,
)

################################################################################
# Workflow class
################################################################################


class PaperResearcher(Workflow):
    """End‑to‑end arXiv literature reviewer."""

    searcher: Agent = searcher
    reader: Agent = reader
    writer: Agent = writer

    def run_workflow(
        self, question: str, pages_to_read: int = 6, use_cache: bool = True
    ) -> Iterator[RunResponse]:
        console.print(
            f"[bold blue]📚 Starting literature review for: {question}[/bold blue]"
        )

        # 1. SEARCH
        console.print(
            "[bold yellow]🔍 Searching arXiv for relevant papers...[/bold yellow]"
        )
        search_out = self._search(question, use_cache)
        if not search_out.papers:
            console.print(
                "[bold red]❌ No papers found matching your query.[/bold red]"
            )
            yield RunResponse(
                event=RunEvent.workflow_completed, content="Search returned no papers."
            )
            return
        console.print(
            f"[bold green]✓ Found {len(search_out.papers)} papers[/bold green]"
        )

        # 2. RANK
        console.print("[bold yellow]📊 Ranking papers by relevance...[/bold yellow]")
        top_ids = self._rank_and_select(question, search_out)
        if not top_ids:
            console.print("[bold red]❌ No papers selected after ranking.[/bold red]")
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Semantic ranking produced no candidates.",
            )
            return
        console.print(
            f"[bold green]✓ Selected {len(top_ids)} most relevant papers[/bold green]"
        )

        # 3. READ
        console.print("[bold yellow]📖 Reading and analyzing papers...[/bold yellow]")
        parsed_papers = self._read_papers(top_ids, pages_to_read, use_cache)
        console.print(
            f"[bold green]✓ Successfully processed {len(parsed_papers)} papers[/bold green]"
        )

        # 4. BUILD RETRIEVER
        console.print("[bold yellow]🔧 Building search index...[/bold yellow]")
        chunks, metas = self._build_chunks(parsed_papers)
        index = self._build_faiss_index(chunks)
        console.print(
            f"[bold green]✓ Created index with {len(chunks)} text chunks[/bold green]"
        )

        embedder = get_embedder()
        q_vec = embedder.encode(question, convert_to_numpy=True).astype("float32")
        _, I = index.search(q_vec.reshape(1, -1), k=6)
        context_chunks = [chunks[i] for i in I[0]]
        console.print(
            "[bold green]✓ Retrieved most relevant text passages[/bold green]"
        )

        # 5. WRITE
        console.print("[bold yellow]✍️ Generating literature review...[/bold yellow]")
        writer_payload = json.dumps(
            {
                "question": question,
                "chunks": context_chunks,
            },
            indent=2,
        )

        yield from self.writer.run(writer_payload, stream=True)
        console.print("[bold green]✓ Literature review completed[/bold green]")

    def _search(self, question: str, use_cache: bool) -> SearchOutput:
        cache = self.session_state.setdefault("search", {})
        if use_cache and question in cache:
            console.print("[bold blue]Using cached search results[/bold blue]")
            return SearchOutput.model_validate(cache[question])

        resp = self.searcher.run(question)
        search_out: SearchOutput = resp.content
        cache[question] = search_out
        return search_out

    def _rank_and_select(
        self,
        query: str,
        search_out: SearchOutput,
        cap: int = 12,
    ) -> List[str]:
        embedder = get_embedder()
        q_vec = embedder.encode(query, convert_to_tensor=True)

        scored: List[Tuple[float, PaperMeta]] = []
        for p in search_out.papers:
            vec = embedder.encode(f"{p.title} {p.summary}", convert_to_tensor=True)
            scored.append((util.pytorch_cos_sim(q_vec, vec).item(), p))

        if not scored:
            return []
        scored.sort(reverse=True, key=lambda x: x[0])

        # dynamic threshold: within 1σ of max
        scores = [s for s, _ in scored]
        max_s = scores[0]
        std_s = (sum((s - max_s) ** 2 for s in scores) / len(scores)) ** 0.5
        keep = [p.id for s, p in scored if s >= max_s - std_s][:cap]
        # logger.info("Selected %d papers after ranking.", len(keep))
        return keep

    def _read_papers(
        self, ids: List[str], pages: int, use_cache: bool
    ) -> List[ParsedPaper]:
        cache_key = (tuple(ids), pages)
        cache = self.session_state.setdefault("parsed", {})
        if use_cache and cache_key in cache:
            console.print("[bold blue]Using cached paper content[/bold blue]")
            return [ParsedPaper.model_validate(p) for p in cache[cache_key]]

        console.print(f"[bold blue]Reading {len(ids)} papers...[/bold blue]")
        resp = self.reader.run(json.dumps({"id_list": ids, "pages_to_read": pages}))

        if isinstance(resp.content, str):
            console.print(
                "[bold yellow]Received string response, attempting to parse as JSON...[/bold yellow]"
            )
            try:
                parsed_data = json.loads(resp.content)
                if "papers" in parsed_data:
                    parsed = ReadOutput(
                        papers=[
                            ParsedPaper.model_validate(p) for p in parsed_data["papers"]
                        ]
                    )
                else:
                    parsed = ReadOutput(
                        papers=[ParsedPaper.model_validate(p) for p in parsed_data]
                    )
            except json.JSONDecodeError:
                console.print(
                    "[bold red]Failed to parse response as JSON. Using fallback method.[/bold red]"
                )
                parsed = ReadOutput(
                    papers=[
                        ParsedPaper(
                            id=ids[0] if ids else "unknown",
                            title="Paper",
                            pages=[
                                PageChunk(
                                    id=ids[0] if ids else "unknown",
                                    page=1,
                                    text=resp.content,
                                )
                            ],
                        )
                    ]
                )
        else:
            try:
                if hasattr(resp.content, "papers"):
                    parsed = resp.content
                else:
                    parsed = ReadOutput(
                        papers=[ParsedPaper.model_validate(p) for p in resp.content]
                    )
            except Exception as e:
                console.print(
                    f"[bold red]Error processing structured response: {e}[/bold red]"
                )
                parsed = ReadOutput(
                    papers=[
                        ParsedPaper(
                            id=ids[0] if ids else "unknown",
                            title="Paper",
                            pages=[
                                PageChunk(
                                    id=ids[0] if ids else "unknown",
                                    page=1,
                                    text=str(resp.content),
                                )
                            ],
                        )
                    ]
                )

        cache[cache_key] = [p.model_dump() for p in parsed.papers]
        return parsed.papers

    def _build_chunks(self, papers: List[ParsedPaper]):
        chunks: List[str] = []
        for pap in papers:
            for pg in pap.pages:
                for chunk in chunk_text(pg.text):
                    chunks.append(chunk)
        return chunks, []  # metas unused in this minimal version

    def _build_faiss_index(self, chunks: List[str]):
        embedder = get_embedder()
        vecs = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        return index


################################################################################
# CLI helper
################################################################################

if __name__ == "__main__":
    from rich.prompt import Prompt

    # Check for DeepSeek API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        console.print(
            "[bold red]Error: DEEPSEEK_API_KEY environment variable not set.[/bold red]"
        )
        console.print(
            "Please set it with: [bold yellow]export DEEPSEEK_API_KEY='your-api-key'[/bold yellow]"
        )
        exit(1)

    question = Prompt.ask(
        "[bold green]Research question?[/]",
        default="How to make low latency text-to-speech?",
    )

    console.print("[bold blue]Initializing research workflow...[/bold blue]")
    wf = PaperResearcher(
        session_id=f"paper-research-{question[:40].lower().replace(' ', '-')}",
        storage=SqliteStorage(
            table_name="paper_researcher", db_file="tmp/agno_workflows.db"
        ),
        debug_mode=True,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        responses = wf.run_workflow(question)
        pprint_run_response(responses, markdown=True)
