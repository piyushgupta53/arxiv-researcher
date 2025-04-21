# arXiv Researcher

An AI-powered tool that helps researchers quickly understand new papers in their field by automatically searching, analyzing, and summarizing arXiv papers.

## Features

- 🔍 **Smart Paper Search**: Searches arXiv for relevant papers based on your research question
- 📊 **Semantic Ranking**: Ranks papers by relevance using advanced embedding techniques
- 📑 **Content Extraction**: Automatically extracts and processes key content from papers
- 📝 **Research Summary**: Generates comprehensive literature reviews with citations
- ⚡ **Efficient**: Processes multiple papers simultaneously and caches results

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/arxiv-researcher.git
cd arxiv-researcher
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your DeepSeek API key:

```bash
export DEEPSEEK_API_KEY='your-api-key'
```

## Usage

Run the main script:

```bash
python main.py
```

You'll be prompted to enter your research question. The tool will then:

1. Search arXiv for relevant papers
2. Rank and select the most relevant ones
3. Extract content from the papers
4. Generate a comprehensive literature review

## Output

The tool generates a structured literature review that includes:

- Executive Summary
- Methodology Analysis
- Comparative Analysis
- Critical Discussion
- Research Gaps and Future Directions
- References with citations

## Requirements

- Python 3.8+
- DeepSeek API key
- Internet connection for arXiv access

## Project Structure

- `main.py`: Main workflow implementation
- `arxiv_tools.py`: Tools for interacting with arXiv API
- `tmp/`: Directory for temporary files and caches
- `venv/`: Virtual environment (not tracked in git)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Agno Workflow](https://github.com/agno-ai/agno)
- Uses [DeepSeek](https://deepseek.ai/) for AI-powered analysis
- Leverages [arXiv API](https://arxiv.org/help/api/) for paper access
