"""
Document Agent - LangChain agent for document operations
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

from app.tools.document_tools import (
    get_document_tools,
    document_store,
    SearchDocumentsTool,
    SummarizeDocumentTool,
    AnswerQuestionTool,
    ListDocumentsTool,
    ExtractEntitiesTool
)
from app.chains.qa_chain import DocumentQAChain, ConversationalQAChain
from app.processors.document_processor import DocumentProcessorFactory


@dataclass
class AgentContext:
    """Maintains agent conversation context."""
    current_document_id: Optional[str] = None
    last_search_query: Optional[str] = None
    last_search_results: List[Dict] = field(default_factory=list)
    conversation_topic: Optional[str] = None
    user_intent: Optional[str] = None

    def update_from_message(self, message: str):
        """Update context based on user message."""
        message_lower = message.lower()

        # Detect intent
        if any(w in message_lower for w in ['search', 'find', 'look for']):
            self.user_intent = "search"
        elif any(w in message_lower for w in ['summarize', 'summary', 'brief']):
            self.user_intent = "summarize"
        elif any(w in message_lower for w in ['extract', 'entities', 'names', 'dates']):
            self.user_intent = "extract"
        elif any(w in message_lower for w in ['compare', 'difference', 'similar']):
            self.user_intent = "compare"
        elif '?' in message:
            self.user_intent = "question"
        else:
            self.user_intent = "general"


class DocumentAgent:
    """
    Intelligent document processing agent that can:
    - Upload and process documents
    - Search through document content
    - Answer questions based on documents
    - Extract entities and key information
    - Summarize documents
    - Compare documents
    """

    SYSTEM_PROMPT = """You are an intelligent document assistant that helps users work with their documents.

Your capabilities include:
1. **Document Processing**: Upload and analyze PDF, Word, text, and markdown files
2. **Smart Search**: Find relevant content across all uploaded documents
3. **Question Answering**: Answer questions based on document content
4. **Summarization**: Generate concise summaries of documents
5. **Entity Extraction**: Extract people, organizations, dates, and other entities
6. **Document Comparison**: Compare two documents to find similarities and differences

Guidelines:
- Always cite your sources when answering questions
- If you're uncertain about an answer, indicate your confidence level
- Ask for clarification when the user's request is ambiguous
- Suggest relevant follow-up questions or actions
- Format responses clearly using markdown when appropriate

You have access to tools for searching, summarizing, and analyzing documents.
Use these tools to provide accurate, helpful responses based on the user's documents."""

    def __init__(self, llm=None, verbose: bool = False):
        """Initialize the document agent."""
        self.llm = llm
        self.verbose = verbose
        self.context = AgentContext()
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = get_document_tools()
        self.qa_chain = ConversationalQAChain(llm=llm)

        # Setup prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Setup memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )

    def upload_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Upload and process a document."""
        try:
            # Process document
            doc = DocumentProcessorFactory.process_document(content, filename)

            # Add to store
            document_store.add_document(doc)

            # Update context
            self.context.current_document_id = doc.document_id

            return {
                "success": True,
                "document_id": doc.document_id,
                "filename": doc.metadata.filename,
                "type": doc.metadata.doc_type.value,
                "word_count": doc.metadata.word_count,
                "chunk_count": len(doc.chunks),
                "message": f"Successfully uploaded '{filename}' ({doc.metadata.word_count} words, {len(doc.chunks)} chunks)"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process document: {str(e)}"
            }

    def chat(self, user_message: str) -> str:
        """
        Process user message and generate response.

        Args:
            user_message: User's input message

        Returns:
            Agent's response string
        """
        # Update context
        self.context.update_from_message(user_message)

        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Generate response based on intent
        response = self._generate_response(user_message)

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        return response

    def _generate_response(self, user_message: str) -> str:
        """Generate response based on user intent and context."""
        intent = self.context.user_intent

        if intent == "search":
            return self._handle_search(user_message)
        elif intent == "summarize":
            return self._handle_summarize(user_message)
        elif intent == "extract":
            return self._handle_extract(user_message)
        elif intent == "compare":
            return self._handle_compare(user_message)
        elif intent == "question":
            return self._handle_question(user_message)
        else:
            return self._handle_general(user_message)

    def _handle_search(self, message: str) -> str:
        """Handle search requests."""
        # Extract search query
        query = message.lower()
        for prefix in ['search for', 'find', 'look for', 'search']:
            if prefix in query:
                query = query.split(prefix, 1)[-1].strip()
                break

        tool = SearchDocumentsTool()
        result = tool._run(query=query, document_id=self.context.current_document_id)
        data = json.loads(result)

        if not data.get("results"):
            return "I couldn't find any matching content in the documents. Try a different search term or upload more documents."

        response = f"**Search Results for '{query}':**\n\n"

        for i, res in enumerate(data["results"], 1):
            response += f"**{i}. From '{res['document']}' (Page {res['page']}):**\n"
            response += f"{res['content']}\n\n"

        response += f"\n*Found {data['count']} relevant sections.*"

        # Store search context
        self.context.last_search_query = query
        self.context.last_search_results = data["results"]

        return response

    def _handle_summarize(self, message: str) -> str:
        """Handle summarization requests."""
        # Check for specific document or use current
        doc_id = self.context.current_document_id

        docs = document_store.list_documents()
        if not docs:
            return "No documents have been uploaded yet. Please upload a document first."

        if not doc_id:
            doc_id = docs[0]["document_id"]

        tool = SummarizeDocumentTool()
        result = tool._run(document_id=doc_id)
        data = json.loads(result)

        if "error" in data:
            return f"Error: {data['error']}"

        response = f"**Summary of '{data['document']}':**\n\n"
        response += data["summary"]
        response += f"\n\n*Original: {data['original_length']} characters | Summary: {data['summary_length']} characters*"

        return response

    def _handle_extract(self, message: str) -> str:
        """Handle entity extraction requests."""
        doc_id = self.context.current_document_id

        docs = document_store.list_documents()
        if not docs:
            return "No documents have been uploaded yet. Please upload a document first."

        if not doc_id:
            doc_id = docs[0]["document_id"]

        tool = ExtractEntitiesTool()
        result = tool._run(document_id=doc_id)
        data = json.loads(result)

        if "error" in data:
            return f"Error: {data['error']}"

        response = f"**Entities Extracted from '{data['document']}':**\n\n"

        entities = data["entities"]

        if entities.get("persons"):
            response += f"**People:** {', '.join(entities['persons'][:10])}\n\n"

        if entities.get("organizations"):
            response += f"**Organizations:** {', '.join(entities['organizations'][:10])}\n\n"

        if entities.get("locations"):
            response += f"**Locations:** {', '.join(entities['locations'][:10])}\n\n"

        if entities.get("dates"):
            response += f"**Dates:** {', '.join(entities['dates'][:10])}\n\n"

        if entities.get("emails"):
            response += f"**Emails:** {', '.join(entities['emails'][:5])}\n\n"

        if entities.get("urls"):
            response += f"**URLs:** {', '.join(entities['urls'][:5])}\n\n"

        return response

    def _handle_compare(self, message: str) -> str:
        """Handle document comparison requests."""
        docs = document_store.list_documents()

        if len(docs) < 2:
            return "I need at least two documents to compare. Please upload more documents."

        # Compare first two documents
        from app.tools.document_tools import CompareDocumentsTool

        tool = CompareDocumentsTool()
        result = tool._run(
            document_id_1=docs[0]["document_id"],
            document_id_2=docs[1]["document_id"]
        )
        data = json.loads(result)

        response = "**Document Comparison:**\n\n"
        response += f"**Document 1:** {data['document_1']['filename']} ({data['document_1']['word_count']} words)\n"
        response += f"**Document 2:** {data['document_2']['filename']} ({data['document_2']['word_count']} words)\n\n"
        response += f"**Similarity Score:** {data['similarity_score'] * 100:.1f}%\n\n"

        if data.get("common_terms"):
            response += f"**Common Terms:** {', '.join(data['common_terms'][:15])}\n\n"

        if data.get("unique_to_doc1"):
            response += f"**Unique to Doc 1:** {', '.join(data['unique_to_doc1'][:10])}\n\n"

        if data.get("unique_to_doc2"):
            response += f"**Unique to Doc 2:** {', '.join(data['unique_to_doc2'][:10])}\n\n"

        return response

    def _handle_question(self, message: str) -> str:
        """Handle question answering."""
        docs = document_store.list_documents()

        if not docs:
            return "No documents have been uploaded yet. Please upload a document first, then ask your question."

        # Use conversational QA chain
        result = self.qa_chain.answer(
            question=message,
            document_id=self.context.current_document_id
        )

        response = f"**Answer:**\n{result.answer}\n\n"

        if result.sources:
            response += "**Sources:**\n"
            for source in result.sources[:3]:
                response += f"- {source['document']} (Page {source['page']})\n"

        response += f"\n*Confidence: {result.confidence * 100:.0f}%*"

        return response

    def _handle_general(self, message: str) -> str:
        """Handle general messages."""
        docs = document_store.list_documents()

        if not docs:
            return (
                "Welcome! I'm your Document Assistant. I can help you:\n\n"
                "- **Upload documents** (PDF, Word, Text, Markdown)\n"
                "- **Search** through document content\n"
                "- **Answer questions** based on your documents\n"
                "- **Summarize** document content\n"
                "- **Extract** key information (people, dates, organizations)\n"
                "- **Compare** two documents\n\n"
                "To get started, please upload a document."
            )

        # List documents and suggest actions
        response = f"You have **{len(docs)} document(s)** uploaded:\n\n"

        for doc in docs[:5]:
            response += f"- **{doc['filename']}** ({doc['type']}, {doc['word_count']} words)\n"

        response += "\n**What would you like to do?**\n"
        response += "- Ask a question about your documents\n"
        response += "- Search for specific content\n"
        response += "- Get a summary\n"
        response += "- Extract entities\n"

        if len(docs) >= 2:
            response += "- Compare documents\n"

        return response

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of uploaded documents."""
        return document_store.list_documents()

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document."""
        success = document_store.delete_document(doc_id)

        if success:
            if self.context.current_document_id == doc_id:
                self.context.current_document_id = None
            return {"success": True, "message": "Document deleted successfully"}

        return {"success": False, "message": "Document not found"}

    def set_active_document(self, doc_id: str) -> Dict[str, Any]:
        """Set the active document for context."""
        doc = document_store.get_document(doc_id)

        if doc:
            self.context.current_document_id = doc_id
            return {
                "success": True,
                "message": f"Active document set to '{doc.metadata.filename}'"
            }

        return {"success": False, "message": "Document not found"}

    def reset(self):
        """Reset agent state."""
        self.context = AgentContext()
        self.conversation_history = []
        self.qa_chain.clear_history()
        self.memory.clear()


def create_document_agent(llm=None, verbose: bool = False) -> DocumentAgent:
    """Factory function to create document agent."""
    return DocumentAgent(llm=llm, verbose=verbose)
