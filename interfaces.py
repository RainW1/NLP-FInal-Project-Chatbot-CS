"""
============================================================
INTERFACE CONTRACTS - E-Commerce Customer Service Chatbot
============================================================

IMPORTANT: Read this file carefully before coding!

This file defines the INPUT/OUTPUT contract for every function.
Everyone MUST follow these contracts so our code integrates smoothly.

Agreed by team on: [DATE]
============================================================
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

# ============================================================
# SHARED DATA TYPES
# ============================================================

@dataclass
class Intent:
    """Classification result for user intent"""
    label: str          # e.g., "complaint_delivery", "order_status", "product_inquiry"
    confidence: float   # 0.0 to 1.0
    
    # Possible intent labels:
    # - "complaint_delivery"  : Keluhan pengiriman
    # - "complaint_product"   : Keluhan produk rusak/salah
    # - "order_status"        : Cek status pesanan
    # - "product_inquiry"     : Tanya info produk
    # - "return_refund"       : Mau return/refund
    # - "greeting"            : Sapaan
    # - "other"               : Lainnya

@dataclass  
class Entities:
    """Extracted entities from user message"""
    order_id: Optional[str] = None      # e.g., "ORDER123", "ORD-456789"
    product_name: Optional[str] = None  # e.g., "iPhone 15", "Sepatu Nike"
    date: Optional[str] = None          # e.g., "kemarin", "5 hari lalu", "2024-01-15"
    amount: Optional[str] = None        # e.g., "Rp 500.000", "2 barang"
    person_name: Optional[str] = None   # e.g., customer name mentioned

@dataclass
class Message:
    """Single message in conversation"""
    role: str       # "user" or "assistant"
    content: str    # The message text

@dataclass
class RetrievedDoc:
    """Document retrieved from RAG"""
    content: str        # The document text
    source: str         # e.g., "faq.txt", "shipping_policy.txt"
    relevance: float    # 0.0 to 1.0


# ============================================================
# PERSON A: SPEECH & CLASSIFICATION
# ============================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Converts speech audio to text using Whisper.
    
    Args:
        audio_bytes: Raw audio data from microphone (WAV format)
        
    Returns:
        Transcribed text as string
        
    Example:
        >>> audio = record_from_microphone()
        >>> text = transcribe_audio(audio)
        >>> print(text)
        "Halo, pesanan saya ORDER123 belum sampai sudah 5 hari"
        
    Owner: Person A
    """
    pass


def classify_intent(text: str) -> Intent:
    """
    Classifies the user's intent from their message.
    
    Args:
        text: User message (string)
        
    Returns:
        Intent object with label and confidence
        
    Example:
        >>> intent = classify_intent("pesanan saya belum sampai")
        >>> print(intent.label)
        "complaint_delivery"
        >>> print(intent.confidence)
        0.92
        
    Owner: Person A
    """
    pass


# ============================================================
# PERSON B: CHATBOT & RAG
# ============================================================

def retrieve_knowledge(query: str, top_k: int = 3) -> List[RetrievedDoc]:
    """
    Retrieves relevant documents from knowledge base using RAG.
    
    Args:
        query: Search query (usually user's message)
        top_k: Number of documents to retrieve
        
    Returns:
        List of RetrievedDoc objects
        
    Example:
        >>> docs = retrieve_knowledge("berapa lama pengiriman")
        >>> print(docs[0].content)
        "Pengiriman standar membutuhkan 3-5 hari kerja..."
        >>> print(docs[0].source)
        "shipping_policy.txt"
        
    Owner: Person B
    """
    pass


def generate_response(
    user_message: str,
    intent: Intent,
    entities: Entities,
    retrieved_docs: List[RetrievedDoc],
    conversation_history: List[Message]
) -> str:
    """
    Generates chatbot response using hybrid approach (rules + LLM).
    
    For simple intents (greeting, FAQ), use rule-based.
    For complex queries, use LLM with RAG context.
    
    Args:
        user_message: Current user message
        intent: Classified intent from classify_intent()
        entities: Extracted entities from extract_entities()
        retrieved_docs: Relevant docs from retrieve_knowledge()
        conversation_history: Previous messages for context
        
    Returns:
        Bot response as string
        
    Example:
        >>> response = generate_response(
        ...     user_message="ORDER123 belum sampai 5 hari",
        ...     intent=Intent("complaint_delivery", 0.95),
        ...     entities=Entities(order_id="ORDER123", date="5 hari"),
        ...     retrieved_docs=[...],
        ...     conversation_history=[...]
        ... )
        >>> print(response)
        "Mohon maaf atas keterlambatan pesanan ORDER123. Kami akan 
         segera mengecek status pengiriman Anda..."
        
    Owner: Person B
    """
    pass


# ============================================================
# PERSON C: NER & SUMMARIZATION
# ============================================================

def extract_entities(text: str) -> Entities:
    """
    Extracts named entities from user message.
    
    Args:
        text: User message (string)
        
    Returns:
        Entities object with extracted information
        
    Example:
        >>> entities = extract_entities("ORDER123 saya beli iPhone kemarin belum sampai")
        >>> print(entities.order_id)
        "ORDER123"
        >>> print(entities.product_name)
        "iPhone"
        >>> print(entities.date)
        "kemarin"
        
    Owner: Person C
    """
    pass


def summarize_conversation(messages: List[Message]) -> str:
    """
    Generates a summary of the conversation for records.
    
    Args:
        messages: List of Message objects (full conversation)
        
    Returns:
        Summary string (2-3 sentences)
        
    Example:
        >>> messages = [
        ...     Message("user", "ORDER123 belum sampai"),
        ...     Message("assistant", "Mohon maaf, kami cek dulu ya"),
        ...     Message("user", "sudah 5 hari lho"),
        ...     Message("assistant", "Baik, kami akan prioritaskan...")
        ... ]
        >>> summary = summarize_conversation(messages)
        >>> print(summary)
        "Pelanggan mengeluhkan keterlambatan pengiriman ORDER123 
         yang sudah 5 hari. CS meminta maaf dan akan memprioritaskan 
         pengecekan status pengiriman."
        
    Owner: Person C
    """
    pass


def compare_models(text_samples: List[str]) -> Dict:
    """
    Compares two summarization models on given samples.
    
    Args:
        text_samples: List of conversation texts to summarize
        
    Returns:
        Dict with comparison metrics (ROUGE scores, latency, etc.)
        
    Example:
        >>> results = compare_models(sample_conversations)
        >>> print(results)
        {
            "model_a": {"name": "BART", "rouge1": 0.45, "rouge2": 0.23, "latency_ms": 120},
            "model_b": {"name": "PEGASUS", "rouge1": 0.48, "rouge2": 0.25, "latency_ms": 150},
            "winner": "PEGASUS"
        }
        
    Owner: Person C
    """
    pass


# ============================================================
# INTEGRATION CHECKLIST
# ============================================================
"""
Before merging, verify:

[ ] All functions return the EXACT types specified above
[ ] No function signature changes without team agreement
[ ] Test your function with the examples in docstrings
[ ] Update this file if interface changes (tell the team!)

Integration flow in app.py:

1. audio = get_audio_from_mic()
2. text = transcribe_audio(audio)           # Person A
3. intent = classify_intent(text)           # Person A
4. entities = extract_entities(text)        # Person C
5. docs = retrieve_knowledge(text)          # Person B
6. response = generate_response(...)        # Person B
7. summary = summarize_conversation(...)    # Person C
"""
