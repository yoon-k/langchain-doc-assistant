"""
Document Tools - LangChain tools for document operations
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from app.processors.document_processor import (
    ProcessedDocument,
    DocumentChunk,
    DocumentProcessorFactory
)


# In-memory document store (in production, use a proper vector database)
class DocumentStore:
    """Simple in-memory document store."""

    def __init__(self):
        self.documents: Dict[str, ProcessedDocument] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def add_document(self, doc: ProcessedDocument) -> str:
        """Add document to store."""
        self.documents[doc.document_id] = doc
        for chunk in doc.chunks:
            self.chunks[chunk.chunk_id] = chunk
        return doc.document_id

    def get_document(self, doc_id: str) -> Optional[ProcessedDocument]:
        """Get document by ID."""
        return self.documents.get(doc_id)

    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Simple keyword search (in production, use vector similarity)."""
        query_terms = query.lower().split()
        results = []

        for chunk_id, chunk in self.chunks.items():
            if doc_id and chunk.document_id != doc_id:
                continue

            content_lower = chunk.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)

            if score > 0:
                results.append((score, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:top_k]]

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        return [
            {
                "document_id": doc.document_id,
                "filename": doc.metadata.filename,
                "type": doc.metadata.doc_type.value,
                "word_count": doc.metadata.word_count,
                "chunk_count": len(doc.chunks)
            }
            for doc in self.documents.values()
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from store."""
        if doc_id in self.documents:
            doc = self.documents[doc_id]
            for chunk in doc.chunks:
                if chunk.chunk_id in self.chunks:
                    del self.chunks[chunk.chunk_id]
            del self.documents[doc_id]
            return True
        return False


# Global store instance
document_store = DocumentStore()


class SearchDocumentsInput(BaseModel):
    """Input for document search."""
    query: str = Field(description="Search query to find relevant content")
    document_id: Optional[str] = Field(default=None, description="Specific document to search in")
    top_k: int = Field(default=5, description="Number of results to return")


class SearchDocumentsTool(BaseTool):
    """Tool for searching documents."""
    name: str = "search_documents"
    description: str = "Search through uploaded documents to find relevant content"
    args_schema: type[BaseModel] = SearchDocumentsInput

    def _run(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5
    ) -> str:
        """Search documents."""
        chunks = document_store.search_chunks(query, top_k, document_id)

        if not chunks:
            return json.dumps({"results": [], "message": "No matching content found"})

        results = []
        for chunk in chunks:
            doc = document_store.get_document(chunk.document_id)
            results.append({
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                "document": doc.metadata.filename if doc else "Unknown",
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id
            })

        return json.dumps({"results": results, "count": len(results)}, indent=2)


class SummarizeDocumentInput(BaseModel):
    """Input for document summarization."""
    document_id: str = Field(description="ID of document to summarize")
    max_length: int = Field(default=500, description="Maximum summary length")


class SummarizeDocumentTool(BaseTool):
    """Tool for summarizing documents."""
    name: str = "summarize_document"
    description: str = "Generate a summary of a document's content"
    args_schema: type[BaseModel] = SummarizeDocumentInput

    def _run(self, document_id: str, max_length: int = 500) -> str:
        """Summarize document."""
        doc = document_store.get_document(document_id)

        if not doc:
            return json.dumps({"error": "Document not found"})

        # Extract key sentences (simplified summarization)
        content = doc.raw_content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Score sentences by keyword frequency
        word_freq = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1

        scored_sentences = []
        for sentence in sentences:
            score = sum(word_freq.get(word.lower(), 0) for word in sentence.split())
            scored_sentences.append((score, sentence))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Build summary
        summary_parts = []
        current_length = 0
        for _, sentence in scored_sentences[:10]:
            if current_length + len(sentence) < max_length:
                summary_parts.append(sentence)
                current_length += len(sentence)

        summary = ". ".join(summary_parts) + "."

        return json.dumps({
            "document": doc.metadata.filename,
            "summary": summary,
            "word_count": doc.metadata.word_count,
            "original_length": len(content),
            "summary_length": len(summary)
        }, indent=2)


class ExtractEntitiesInput(BaseModel):
    """Input for entity extraction."""
    document_id: str = Field(description="ID of document to extract entities from")
    entity_types: List[str] = Field(
        default=["person", "organization", "location", "date"],
        description="Types of entities to extract"
    )


class ExtractEntitiesTool(BaseTool):
    """Tool for extracting named entities from documents."""
    name: str = "extract_entities"
    description: str = "Extract named entities (people, organizations, locations, dates) from a document"
    args_schema: type[BaseModel] = ExtractEntitiesInput

    def _run(
        self,
        document_id: str,
        entity_types: List[str] = None
    ) -> str:
        """Extract entities from document."""
        doc = document_store.get_document(document_id)

        if not doc:
            return json.dumps({"error": "Document not found"})

        content = doc.raw_content
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "emails": [],
            "urls": [],
            "phone_numbers": []
        }

        # Email pattern
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content)
        entities["emails"] = list(set(emails))

        # URL pattern
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        entities["urls"] = list(set(urls))

        # Date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, content)
            entities["dates"].extend(dates)
        entities["dates"] = list(set(entities["dates"]))

        # Phone patterns
        phones = re.findall(r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', content)
        entities["phone_numbers"] = list(set(phones))

        # Simple capitalized word extraction for names/orgs (simplified NER)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        # Filter common words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'Here', 'There', 'When', 'Where', 'What', 'How', 'Why'}
        capitalized = [w for w in capitalized if w not in stop_words and len(w) > 2]

        # Heuristic: 2 words likely person, 1 word or with Inc/Corp likely org
        for phrase in set(capitalized):
            if any(kw in phrase for kw in ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Group']):
                entities["organizations"].append(phrase)
            elif len(phrase.split()) == 2:
                entities["persons"].append(phrase)
            elif len(phrase.split()) == 1 and phrase[0].isupper():
                entities["locations"].append(phrase)

        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))[:20]  # Limit to 20 each

        return json.dumps({
            "document": doc.metadata.filename,
            "entities": entities
        }, indent=2)


class AnswerQuestionInput(BaseModel):
    """Input for question answering."""
    question: str = Field(description="Question to answer based on document content")
    document_id: Optional[str] = Field(default=None, description="Specific document to answer from")


class AnswerQuestionTool(BaseTool):
    """Tool for answering questions based on documents."""
    name: str = "answer_question"
    description: str = "Answer questions based on uploaded document content"
    args_schema: type[BaseModel] = AnswerQuestionInput

    def _run(
        self,
        question: str,
        document_id: Optional[str] = None
    ) -> str:
        """Answer question from documents."""
        # Search for relevant chunks
        chunks = document_store.search_chunks(question, top_k=3, doc_id=document_id)

        if not chunks:
            return json.dumps({
                "answer": "I couldn't find relevant information to answer this question.",
                "confidence": 0.0,
                "sources": []
            })

        # Combine relevant content
        context = "\n\n".join([chunk.content for chunk in chunks])

        # Simple extractive answer (in production, use LLM)
        # Find sentences containing question keywords
        question_words = [w.lower() for w in question.split() if len(w) > 3]
        sentences = re.split(r'[.!?]+', context)

        best_sentences = []
        for sentence in sentences:
            score = sum(1 for w in question_words if w in sentence.lower())
            if score > 0:
                best_sentences.append((score, sentence.strip()))

        best_sentences.sort(key=lambda x: x[0], reverse=True)

        if best_sentences:
            answer = ". ".join([s for _, s in best_sentences[:2]]) + "."
            confidence = min(1.0, best_sentences[0][0] / len(question_words))
        else:
            answer = "I found related content but couldn't extract a specific answer."
            confidence = 0.3

        sources = []
        for chunk in chunks:
            doc = document_store.get_document(chunk.document_id)
            sources.append({
                "document": doc.metadata.filename if doc else "Unknown",
                "page": chunk.page_number,
                "excerpt": chunk.content[:200] + "..."
            })

        return json.dumps({
            "answer": answer,
            "confidence": round(confidence, 2),
            "sources": sources
        }, indent=2)


class ListDocumentsInput(BaseModel):
    """Input for listing documents."""
    pass


class ListDocumentsTool(BaseTool):
    """Tool for listing all documents."""
    name: str = "list_documents"
    description: str = "List all uploaded documents"
    args_schema: type[BaseModel] = ListDocumentsInput

    def _run(self) -> str:
        """List all documents."""
        docs = document_store.list_documents()

        if not docs:
            return json.dumps({
                "documents": [],
                "message": "No documents uploaded yet"
            })

        return json.dumps({
            "documents": docs,
            "total_count": len(docs)
        }, indent=2)


class GetDocumentInfoInput(BaseModel):
    """Input for getting document info."""
    document_id: str = Field(description="ID of document to get info about")


class GetDocumentInfoTool(BaseTool):
    """Tool for getting document information."""
    name: str = "get_document_info"
    description: str = "Get detailed information about a specific document"
    args_schema: type[BaseModel] = GetDocumentInfoInput

    def _run(self, document_id: str) -> str:
        """Get document info."""
        doc = document_store.get_document(document_id)

        if not doc:
            return json.dumps({"error": "Document not found"})

        return json.dumps({
            "document_id": doc.document_id,
            "filename": doc.metadata.filename,
            "type": doc.metadata.doc_type.value,
            "size_bytes": doc.metadata.size_bytes,
            "page_count": doc.metadata.page_count,
            "word_count": doc.metadata.word_count,
            "char_count": doc.metadata.char_count,
            "chunk_count": len(doc.chunks),
            "has_tables": len(doc.tables) > 0,
            "table_count": len(doc.tables),
            "created_at": doc.metadata.created_at.isoformat()
        }, indent=2)


class CompareDocumentsInput(BaseModel):
    """Input for comparing documents."""
    document_id_1: str = Field(description="First document ID")
    document_id_2: str = Field(description="Second document ID")


class CompareDocumentsTool(BaseTool):
    """Tool for comparing two documents."""
    name: str = "compare_documents"
    description: str = "Compare two documents to find similarities and differences"
    args_schema: type[BaseModel] = CompareDocumentsInput

    def _run(self, document_id_1: str, document_id_2: str) -> str:
        """Compare two documents."""
        doc1 = document_store.get_document(document_id_1)
        doc2 = document_store.get_document(document_id_2)

        if not doc1 or not doc2:
            return json.dumps({"error": "One or both documents not found"})

        # Get word sets
        words1 = set(doc1.raw_content.lower().split())
        words2 = set(doc2.raw_content.lower().split())

        # Calculate overlap
        common_words = words1 & words2
        only_in_1 = words1 - words2
        only_in_2 = words2 - words1

        # Filter to meaningful words
        common_meaningful = [w for w in common_words if len(w) > 4][:20]
        unique_1 = [w for w in only_in_1 if len(w) > 4][:10]
        unique_2 = [w for w in only_in_2 if len(w) > 4][:10]

        similarity = len(common_words) / max(len(words1 | words2), 1)

        return json.dumps({
            "document_1": {
                "filename": doc1.metadata.filename,
                "word_count": doc1.metadata.word_count
            },
            "document_2": {
                "filename": doc2.metadata.filename,
                "word_count": doc2.metadata.word_count
            },
            "similarity_score": round(similarity, 3),
            "common_terms": common_meaningful,
            "unique_to_doc1": unique_1,
            "unique_to_doc2": unique_2
        }, indent=2)


def get_document_tools() -> List[BaseTool]:
    """Get all document tools."""
    return [
        SearchDocumentsTool(),
        SummarizeDocumentTool(),
        ExtractEntitiesTool(),
        AnswerQuestionTool(),
        ListDocumentsTool(),
        GetDocumentInfoTool(),
        CompareDocumentsTool(),
    ]
