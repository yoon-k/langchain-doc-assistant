# Smart Document Assistant

LangChain-powered intelligent document processing and Q&A system.

## Live Demo

[**View Demo**](https://yoon-k.github.io/langchain-doc-assistant/)

## Features

- **Multi-Format Processing**: PDF, DOCX, TXT, Markdown, CSV support
- **Intelligent Search**: Semantic and keyword-based document search
- **Question Answering**: RAG-based Q&A with source citations
- **Summarization**: Extractive document summarization
- **Entity Extraction**: Extract people, organizations, dates, emails, URLs
- **Document Comparison**: Compare documents for similarities and differences

## Architecture

```
langchain-doc-assistant/
├── app/
│   ├── agents/
│   │   └── document_agent.py      # Main LangChain agent
│   ├── chains/
│   │   └── qa_chain.py            # Question answering chains
│   ├── processors/
│   │   └── document_processor.py  # Document parsing and chunking
│   ├── tools/
│   │   └── document_tools.py      # Custom LangChain tools
│   └── api.py                     # Flask API
├── static/
├── templates/
└── docs/                          # GitHub Pages demo
```

## LangChain Components

### Document Processors
- `TextDocumentProcessor`: Plain text files
- `MarkdownProcessor`: Markdown with section extraction
- `PDFProcessor`: PDF with page-aware chunking
- `DocxProcessor`: Word documents with table extraction
- `CSVProcessor`: Tabular data processing

### Custom Tools
```python
from app.tools.document_tools import get_document_tools

tools = get_document_tools()
# SearchDocumentsTool - Search across documents
# SummarizeDocumentTool - Generate summaries
# ExtractEntitiesTool - Extract named entities
# AnswerQuestionTool - Answer questions
# ListDocumentsTool - List uploaded documents
# CompareDocumentsTool - Compare two documents
```

### QA Chains
```python
from app.chains.qa_chain import create_qa_chain

# Simple extractive QA
qa = create_qa_chain(chain_type="simple")
result = qa.answer("What are the main conclusions?")

# Multi-hop QA for complex questions
qa = create_qa_chain(chain_type="multi_hop")
result = qa.answer("Compare the findings from section 2 and section 5")

# Conversational QA with history
qa = create_qa_chain(chain_type="conversational")
qa.answer("What is the budget?")
qa.answer("Who approved it?")  # Resolves 'it' to 'budget'
```

### Agent Usage
```python
from app.agents.document_agent import create_document_agent

agent = create_document_agent()

# Upload document
with open("report.pdf", "rb") as f:
    agent.upload_document(f.read(), "report.pdf")

# Chat with the agent
response = agent.chat("Summarize the main findings")
response = agent.chat("Who are the key stakeholders mentioned?")
response = agent.chat("Search for budget information")
```

## Installation

```bash
git clone https://github.com/yoon-k/langchain-doc-assistant.git
cd langchain-doc-assistant

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Optional: Set OpenAI API key for LLM features
export OPENAI_API_KEY=your_key_here

python -m app.api
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with the document assistant |
| `/api/documents/upload` | POST | Upload a document |
| `/api/documents` | GET | List all documents |
| `/api/documents/<id>` | DELETE | Delete a document |

## Supported File Types

| Type | Extension | Features |
|------|-----------|----------|
| PDF | .pdf | Page extraction, metadata |
| Word | .docx | Paragraphs, tables |
| Text | .txt | Full text |
| Markdown | .md | Sections, headers |
| CSV | .csv | Row-based chunking |

## Tech Stack

- **LangChain**: Agent and tool framework
- **PyPDF**: PDF processing
- **python-docx**: Word document parsing
- **ChromaDB**: Vector storage (optional)
- **Sentence Transformers**: Embeddings
- **Flask**: API server

## License

MIT License
