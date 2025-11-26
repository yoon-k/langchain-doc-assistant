"""
Question Answering Chain - Advanced QA with retrieval
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from app.processors.document_processor import DocumentChunk, ProcessedDocument
from app.tools.document_tools import document_store


@dataclass
class QAResult:
    """Question answering result."""
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning: Optional[str] = None


class DocumentQAChain:
    """Chain for document-based question answering."""

    QA_PROMPT = """Answer the question based ONLY on the following context.
If you cannot answer from the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, concise answer
2. Quote relevant parts of the context when appropriate
3. If the answer is uncertain, indicate your confidence level
4. List which sources you used

Answer:"""

    def __init__(self, llm=None):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(self.QA_PROMPT)

    def answer(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: int = 5
    ) -> QAResult:
        """Answer a question using document context."""

        # Retrieve relevant chunks
        chunks = document_store.search_chunks(question, top_k=top_k, doc_id=document_id)

        if not chunks:
            return QAResult(
                question=question,
                answer="No relevant documents found to answer this question.",
                confidence=0.0,
                sources=[]
            )

        # Build context from chunks
        context_parts = []
        sources = []

        for chunk in chunks:
            doc = document_store.get_document(chunk.document_id)
            source_info = {
                "document": doc.metadata.filename if doc else "Unknown",
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id
            }
            sources.append(source_info)

            context_parts.append(
                f"[Source: {source_info['document']}, Page {source_info['page']}]\n{chunk.content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        if self.llm:
            # Use LLM for sophisticated answer
            formatted_prompt = self.prompt.format(context=context, question=question)
            answer = self.llm.invoke(formatted_prompt)
            confidence = 0.8
        else:
            # Extractive answer without LLM
            answer, confidence = self._extractive_answer(question, context)

        return QAResult(
            question=question,
            answer=answer,
            confidence=confidence,
            sources=sources
        )

    def _extractive_answer(self, question: str, context: str) -> tuple:
        """Generate extractive answer without LLM."""

        # Extract question keywords
        question_words = [w.lower() for w in question.split() if len(w) > 3]

        # Handle different question types
        if any(w in question.lower() for w in ['what is', 'what are', 'define']):
            return self._answer_definition(question_words, context)
        elif any(w in question.lower() for w in ['how many', 'how much', 'count']):
            return self._answer_quantity(question_words, context)
        elif any(w in question.lower() for w in ['when', 'what time', 'what date']):
            return self._answer_temporal(context)
        elif any(w in question.lower() for w in ['where', 'location', 'place']):
            return self._answer_location(context)
        elif any(w in question.lower() for w in ['why', 'reason', 'because']):
            return self._answer_reason(question_words, context)
        elif any(w in question.lower() for w in ['who', 'person', 'people']):
            return self._answer_person(context)
        else:
            return self._answer_general(question_words, context)

    def _answer_definition(self, keywords: List[str], context: str) -> tuple:
        """Answer definition questions."""
        sentences = re.split(r'[.!?]+', context)

        # Look for definition patterns
        definition_patterns = [
            r'(.+?)\s+is\s+(.+)',
            r'(.+?)\s+refers to\s+(.+)',
            r'(.+?)\s+means\s+(.+)',
            r'(.+?)\s+are\s+(.+)',
        ]

        for sentence in sentences:
            for pattern in definition_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    # Check if keywords are in the sentence
                    if any(kw in sentence.lower() for kw in keywords):
                        return sentence.strip(), 0.7

        # Fallback to most relevant sentence
        return self._find_most_relevant_sentence(keywords, sentences)

    def _answer_quantity(self, keywords: List[str], context: str) -> tuple:
        """Answer quantity questions."""
        sentences = re.split(r'[.!?]+', context)

        # Look for numbers
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', sentence)
                if numbers:
                    return sentence.strip(), 0.7

        return self._find_most_relevant_sentence(keywords, sentences)

    def _answer_temporal(self, context: str) -> tuple:
        """Answer time-related questions."""
        # Look for date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            r'\d{4}',
        ]

        sentences = re.split(r'[.!?]+', context)

        for sentence in sentences:
            for pattern in date_patterns:
                if re.search(pattern, sentence):
                    return sentence.strip(), 0.6

        return "I couldn't find specific date information in the documents.", 0.3

    def _answer_location(self, context: str) -> tuple:
        """Answer location questions."""
        sentences = re.split(r'[.!?]+', context)

        location_indicators = ['in', 'at', 'located', 'based', 'headquarters', 'office']

        for sentence in sentences:
            if any(ind in sentence.lower() for ind in location_indicators):
                # Check for capitalized words (likely locations)
                caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
                if caps:
                    return sentence.strip(), 0.6

        return "I couldn't find specific location information in the documents.", 0.3

    def _answer_reason(self, keywords: List[str], context: str) -> tuple:
        """Answer 'why' questions."""
        sentences = re.split(r'[.!?]+', context)

        reason_indicators = ['because', 'due to', 'since', 'as a result', 'therefore', 'thus']

        for sentence in sentences:
            if any(ind in sentence.lower() for ind in reason_indicators):
                if any(kw in sentence.lower() for kw in keywords):
                    return sentence.strip(), 0.6

        return self._find_most_relevant_sentence(keywords, sentences)

    def _answer_person(self, context: str) -> tuple:
        """Answer 'who' questions."""
        # Look for person names (two capitalized words)
        persons = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', context)

        if persons:
            # Find sentence containing first person mentioned
            sentences = re.split(r'[.!?]+', context)
            for sentence in sentences:
                if persons[0] in sentence:
                    return sentence.strip(), 0.6

        return "I couldn't identify specific people in the documents.", 0.3

    def _answer_general(self, keywords: List[str], context: str) -> tuple:
        """General answer extraction."""
        sentences = re.split(r'[.!?]+', context)
        return self._find_most_relevant_sentence(keywords, sentences)

    def _find_most_relevant_sentence(
        self,
        keywords: List[str],
        sentences: List[str]
    ) -> tuple:
        """Find most relevant sentence based on keyword overlap."""
        scored = []

        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            score = sum(1 for kw in keywords if kw in sentence.lower())
            if score > 0:
                scored.append((score, sentence.strip()))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0]
            confidence = min(0.8, best[0] / max(len(keywords), 1))
            return best[1], confidence

        return "I couldn't find a specific answer in the documents.", 0.2


class MultiHopQAChain:
    """Chain for multi-hop question answering."""

    def __init__(self, llm=None):
        self.llm = llm
        self.base_qa = DocumentQAChain(llm=llm)

    def answer(
        self,
        question: str,
        document_id: Optional[str] = None,
        max_hops: int = 3
    ) -> QAResult:
        """Answer complex questions with multiple retrieval steps."""

        # Decompose question into sub-questions
        sub_questions = self._decompose_question(question)

        all_sources = []
        intermediate_answers = []

        # Answer each sub-question
        for sub_q in sub_questions[:max_hops]:
            result = self.base_qa.answer(sub_q, document_id)
            intermediate_answers.append({
                "question": sub_q,
                "answer": result.answer
            })
            all_sources.extend(result.sources)

        # Synthesize final answer
        if len(intermediate_answers) > 1:
            final_answer = self._synthesize_answer(question, intermediate_answers)
        else:
            final_answer = intermediate_answers[0]["answer"] if intermediate_answers else "No answer found."

        # Deduplicate sources
        unique_sources = []
        seen = set()
        for source in all_sources:
            key = f"{source['document']}_{source['page']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)

        return QAResult(
            question=question,
            answer=final_answer,
            confidence=0.7 if intermediate_answers else 0.2,
            sources=unique_sources,
            reasoning=str(intermediate_answers) if len(intermediate_answers) > 1 else None
        )

    def _decompose_question(self, question: str) -> List[str]:
        """Decompose complex question into simpler sub-questions."""
        sub_questions = [question]

        # Check for compound questions
        if ' and ' in question.lower():
            parts = question.split(' and ')
            if len(parts) >= 2:
                sub_questions = [p.strip() + '?' for p in parts]

        # Check for comparison questions
        elif 'compare' in question.lower() or 'difference' in question.lower():
            # Extract entities being compared
            match = re.search(r'between\s+(.+?)\s+and\s+(.+?)[\?\.]?$', question, re.IGNORECASE)
            if match:
                entity1, entity2 = match.groups()
                sub_questions = [
                    f"What is {entity1.strip()}?",
                    f"What is {entity2.strip()}?",
                    question
                ]

        return sub_questions

    def _synthesize_answer(
        self,
        original_question: str,
        intermediate_answers: List[Dict[str, str]]
    ) -> str:
        """Synthesize final answer from intermediate answers."""

        # Simple concatenation synthesis
        parts = []
        for ia in intermediate_answers:
            if "I couldn't find" not in ia["answer"]:
                parts.append(ia["answer"])

        if parts:
            return " Additionally, ".join(parts)
        else:
            return "I couldn't find enough information to fully answer this question."


class ConversationalQAChain:
    """Chain for conversational question answering with memory."""

    def __init__(self, llm=None):
        self.llm = llm
        self.base_qa = DocumentQAChain(llm=llm)
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 5

    def answer(
        self,
        question: str,
        document_id: Optional[str] = None
    ) -> QAResult:
        """Answer question with conversation context."""

        # Resolve pronouns and references
        resolved_question = self._resolve_references(question)

        # Get answer
        result = self.base_qa.answer(resolved_question, document_id)

        # Update history
        self.conversation_history.append({
            "question": question,
            "resolved_question": resolved_question,
            "answer": result.answer
        })

        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return result

    def _resolve_references(self, question: str) -> str:
        """Resolve pronouns and references using conversation history."""

        if not self.conversation_history:
            return question

        # Common reference patterns
        reference_words = ['it', 'this', 'that', 'they', 'them', 'these', 'those']

        question_lower = question.lower()

        # Check if question contains references
        has_reference = any(f' {ref} ' in f' {question_lower} ' for ref in reference_words)

        if not has_reference:
            return question

        # Get the most recent topic from history
        recent = self.conversation_history[-1]
        recent_question = recent.get("resolved_question", recent.get("question", ""))

        # Extract potential noun phrases from recent question
        # Simple heuristic: get capitalized words or quoted phrases
        nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', recent_question)

        if nouns:
            # Replace 'it' with the first noun found
            topic = nouns[0]
            resolved = question

            for ref in reference_words:
                pattern = rf'\b{ref}\b'
                resolved = re.sub(pattern, topic, resolved, flags=re.IGNORECASE)

            return resolved

        return question

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()


def create_qa_chain(llm=None, chain_type: str = "simple") -> Any:
    """Factory function to create QA chains."""
    chains = {
        "simple": DocumentQAChain,
        "multi_hop": MultiHopQAChain,
        "conversational": ConversationalQAChain
    }

    chain_class = chains.get(chain_type, DocumentQAChain)
    return chain_class(llm=llm)
