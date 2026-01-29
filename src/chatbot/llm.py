"""
Person B: Hybrid Chatbot Module (Rules + LLM)
=============================================

COMPLETE IMPLEMENTATION with:
- Groq API (Llama3-8B)
- Multilingual support (auto-detect language)
- Conversation memory
- Rule-based for simple intents
- RAG context integration

SETUP:
pip install groq

Get API key: https://console.groq.com
"""

import os
import random
from typing import List, Optional, Dict

# ============================================================
# IMPORTS
# ============================================================

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Please install: pip install groq")


# ============================================================
# DATA CLASSES (compatible with interfaces.py)
# ============================================================

class Intent:
    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

class Entities:
    def __init__(self, order_id=None, product_name=None, date=None, amount=None, person_name=None):
        self.order_id = order_id
        self.product_name = product_name
        self.date = date
        self.amount = amount
        self.person_name = person_name

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class RetrievedDoc:
    def __init__(self, content: str, source: str, relevance: float):
        self.content = content
        self.source = source
        self.relevance = relevance


# ============================================================
# CONFIGURATION
# ============================================================

# API Key - Set via environment variable or directly here
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")  # <-- Replace!

# Model selection
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and good quality

# Response settings
MAX_TOKENS = 500
TEMPERATURE = 0.7

# Memory settings
MAX_HISTORY_LENGTH = 20  # Keep last 20 messages


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """Manages conversation history for context"""
    
    def __init__(self, max_length: int = MAX_HISTORY_LENGTH):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
    
    def add(self, role: str, content: str):
        """Add message to history"""
        self.history.append({"role": role, "content": content})
        
        # Trim if too long
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]
    
    def get_history(self, last_n: int = 10) -> List[Dict[str, str]]:
        """Get last N messages"""
        return self.history[-last_n:]
    
    def clear(self):
        """Clear all history"""
        self.history = []
    
    def __len__(self):
        return len(self.history)


# Global memory instance
_memory = ConversationMemory()


def get_memory() -> ConversationMemory:
    """Get global memory instance"""
    return _memory


def clear_memory():
    """Clear conversation memory"""
    _memory.clear()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns: 'id' (Indonesian), 'en' (English), or 'other'
    """
    text_lower = text.lower()
    
    # Indonesian indicators
    id_words = [
        "aku", "saya", "mau", "bisa", "tolong", "terima kasih", "halo",
        "gimana", "bagaimana", "apa", "dimana", "kapan", "sudah", "belum",
        "dong", "ya", "gak", "tidak", "ini", "itu", "kamu", "anda",
        "pesanan", "barang", "kirim", "sampai", "lama", "refund", "komplain"
    ]
    
    # English indicators
    en_words = [
        "i", "want", "can", "please", "thank", "hello", "hi", "how",
        "what", "where", "when", "help", "my", "the", "is", "are",
        "order", "delivery", "refund", "return", "product", "shipping"
    ]
    
    id_count = sum(1 for w in id_words if w in text_lower)
    en_count = sum(1 for w in en_words if w in text_lower)
    
    if id_count > en_count:
        return "id"
    elif en_count > id_count:
        return "en"
    else:
        return "other"


# ============================================================
# RULE-BASED RESPONSES (Multilingual)
# ============================================================

RULE_BASED_RESPONSES = {
    "greeting": {
        "id": [
            "Halo! 👋 Selamat datang di Customer Service kami. Ada yang bisa saya bantu?",
            "Hi! Terima kasih sudah menghubungi kami. Silakan sampaikan pertanyaan Anda 😊",
            "Selamat datang! Saya siap membantu Anda. Ada yang bisa saya bantu?",
        ],
        "en": [
            "Hello! 👋 Welcome to our Customer Service. How can I help you today?",
            "Hi! Thank you for contacting us. How can I assist you? 😊",
            "Welcome! I'm here to help. What can I do for you?",
        ],
        "other": [
            "Hello! 👋 How can I help you today?",
        ]
    },
    "thanks": {
        "id": [
            "Sama-sama! 😊 Senang bisa membantu. Ada lagi yang bisa saya bantu?",
            "Terima kasih kembali! Jangan ragu hubungi kami lagi ya.",
            "You're welcome! Semoga harinya menyenangkan 🌟",
        ],
        "en": [
            "You're welcome! 😊 Happy to help. Anything else I can assist with?",
            "No problem! Feel free to reach out again if you need help.",
            "Glad I could help! Have a great day 🌟",
        ],
        "other": [
            "You're welcome! 😊 Anything else I can help with?",
        ]
    },
    "goodbye": {
        "id": [
            "Terima kasih sudah menghubungi kami. Sampai jumpa! 👋",
            "Baik, semoga harinya menyenangkan! Jangan ragu hubungi kami lagi 😊",
        ],
        "en": [
            "Thank you for contacting us. Goodbye! 👋",
            "Have a great day! Feel free to reach out anytime 😊",
        ],
        "other": [
            "Goodbye! Have a great day! 👋",
        ]
    },
}


def get_rule_based_response(intent: Intent, language: str) -> Optional[str]:
    """
    Get rule-based response if available.
    Returns None if LLM should handle the query.
    """
    if intent.label in RULE_BASED_RESPONSES and intent.confidence > 0.85:
        responses = RULE_BASED_RESPONSES[intent.label]
        lang_responses = responses.get(language, responses.get("other", responses.get("en")))
        return random.choice(lang_responses)
    return None


# ============================================================
# GROQ LLM INTEGRATION
# ============================================================

def create_groq_client() -> Optional[Groq]:
    """Create Groq client with API key"""
    if not GROQ_AVAILABLE:
        print("⚠️ Groq not available. Install with: pip install groq")
        return None
    
    if GROQ_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ Please set your GROQ_API_KEY!")
        return None
    
    return Groq(api_key=GROQ_API_KEY)


def build_system_prompt(
    intent: Intent, 
    entities: Entities, 
    retrieved_docs: List[RetrievedDoc],
    language: str
) -> str:
    """Build system prompt with RAG context and entity info"""
    
    # Build context from retrieved documents
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(f"[{doc.source}]: {doc.content}")
    context = "\n".join(context_parts) if context_parts else "No specific information available."
    
    # Build entity info
    entity_info = []
    if entities.order_id:
        entity_info.append(f"- Order Number: {entities.order_id}")
    if entities.product_name:
        entity_info.append(f"- Product: {entities.product_name}")
    if entities.date:
        entity_info.append(f"- Date/Duration: {entities.date}")
    if entities.amount:
        entity_info.append(f"- Amount: {entities.amount}")
    if entities.person_name:
        entity_info.append(f"- Customer Name: {entities.person_name}")
    
    entity_text = "\n".join(entity_info) if entity_info else "No specific entities detected."
    
    system_prompt = f"""You are a friendly and professional e-commerce customer service assistant named "CS Assistant".

═══════════════════════════════════════════════════════════════
CRITICAL LANGUAGE INSTRUCTION:
Always respond in the SAME LANGUAGE as the user's message!
- User writes Indonesian → Respond in Indonesian
- User writes English → Respond in English
- User writes other language → Respond in that language
Detected language: {language}
═══════════════════════════════════════════════════════════════

CONVERSATION MEMORY:
- You have full memory of this conversation
- If user refers to "my order", "it", "that" without details → check previous messages
- NEVER ask for information the user already provided
- Reference previous context naturally

KNOWLEDGE BASE INFORMATION:
{context}

DETECTED ENTITIES (from current message):
{entity_text}

DETECTED INTENT: {intent.label} (confidence: {intent.confidence:.0%})

RESPONSE GUIDELINES:
1. Be friendly, warm, and professional
2. Keep responses concise (2-4 sentences max)
3. Use 1-2 appropriate emojis
4. Provide clear, actionable solutions
5. If order number exists, mention it for confirmation
6. If you don't know something, be honest and offer alternatives
7. Adapt cultural tone to language (formal for some, casual for Indonesian)

DO NOT:
- Make up information not in knowledge base
- Give long, repetitive answers
- Forget previous context
- Ask for info already provided"""

    return system_prompt


def generate_llm_response(
    user_message: str,
    system_prompt: str,
    memory: ConversationMemory
) -> str:
    """Generate response using Groq LLM with conversation history"""
    
    client = create_groq_client()
    if client is None:
        return fallback_response(user_message)
    
    try:
        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in memory.get_history(last_n=10):
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Groq API
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"⚠️ Groq API error: {e}")
        return fallback_response(user_message)


def fallback_response(user_message: str) -> str:
    """Fallback response when LLM is not available"""
    lang = detect_language(user_message)
    
    if lang == "id":
        return """Mohon maaf, sistem kami sedang mengalami gangguan teknis. 🙏

Silakan coba lagi dalam beberapa saat atau hubungi hotline kami di 021-12345678.

Terima kasih atas kesabarannya."""
    else:
        return """We apologize, our system is currently experiencing technical issues. 🙏

Please try again in a few moments or contact our hotline at 021-12345678.

Thank you for your patience."""


# ============================================================
# MAIN FUNCTION (matches interface contract)
# ============================================================

def generate_response(
    user_message: str,
    intent: Intent,
    entities: Entities,
    retrieved_docs: List[RetrievedDoc],
    conversation_history: List[Message] = None
) -> str:
    """
    Generates chatbot response using hybrid approach (rules + LLM).
    
    Features:
    - Multilingual (auto-detects and responds in same language)
    - Conversation memory (remembers context)
    - Rule-based for simple intents (instant)
    - LLM with RAG for complex queries
    
    Args:
        user_message: Current user message
        intent: Classified intent from classify_intent()
        entities: Extracted entities from extract_entities()
        retrieved_docs: Relevant docs from retrieve_knowledge()
        conversation_history: Previous messages (optional, uses internal memory if not provided)
        
    Returns:
        Bot response as string
    """
    
    # Get memory instance
    memory = get_memory()
    
    # Detect language
    language = detect_language(user_message)
    
    # Step 1: Try rule-based response for simple intents
    rule_response = get_rule_based_response(intent, language)
    if rule_response:
        # Save to memory
        memory.add("user", user_message)
        memory.add("assistant", rule_response)
        return rule_response
    
    # Step 2: Build system prompt with context
    system_prompt = build_system_prompt(intent, entities, retrieved_docs, language)
    
    # Step 3: Generate LLM response
    response = generate_llm_response(user_message, system_prompt, memory)
    
    # Step 4: Save to memory
    memory.add("user", user_message)
    memory.add("assistant", response)
    
    return response


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def test_groq_connection() -> bool:
    """Test if Groq API is working"""
    client = create_groq_client()
    if client is None:
        return False
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Hi, respond with just 'OK'"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


def get_conversation_summary() -> str:
    """Get a summary of current conversation for debugging"""
    memory = get_memory()
    return f"Conversation has {len(memory)} messages in memory."


def reset_conversation():
    """Reset conversation (clear memory)"""
    clear_memory()
    return "Conversation reset."


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hybrid Chatbot (Multilingual + Memory)")
    print("=" * 60)
    
    # Clear memory for fresh test
    clear_memory()
    
    # Test 1: Indonesian greeting
    print("\n🧪 Test 1: Indonesian Greeting")
    response = generate_response(
        user_message="Halo selamat pagi",
        intent=Intent(label="greeting", confidence=0.95),
        entities=Entities(),
        retrieved_docs=[],
    )
    print(f"User: Halo selamat pagi")
    print(f"Bot: {response}")
    
    # Test 2: Indonesian complaint
    print("\n🧪 Test 2: Indonesian Complaint")
    response = generate_response(
        user_message="Pesanan ORDER123 saya belum sampai sudah 5 hari",
        intent=Intent(label="complaint_delivery", confidence=0.88),
        entities=Entities(order_id="ORDER123", date="5 hari"),
        retrieved_docs=[
            RetrievedDoc(
                content="Pengiriman standar membutuhkan 3-5 hari kerja untuk Pulau Jawa.",
                source="shipping_policy",
                relevance=0.85
            )
        ],
    )
    print(f"User: Pesanan ORDER123 saya belum sampai sudah 5 hari")
    print(f"Bot: {response}")
    
    # Test 3: Follow-up (test memory)
    print("\n🧪 Test 3: Follow-up (Testing Memory)")
    response = generate_response(
        user_message="Terus gimana solusinya?",
        intent=Intent(label="other", confidence=0.70),
        entities=Entities(),
        retrieved_docs=[
            RetrievedDoc(
                content="Jika pesanan belum sampai, silakan hubungi CS dengan nomor pesanan.",
                source="faq",
                relevance=0.80
            )
        ],
    )
    print(f"User: Terus gimana solusinya?")
    print(f"Bot: {response}")
    
    # Test 4: English query
    print("\n🧪 Test 4: English Query")
    response = generate_response(
        user_message="How can I get a refund for my damaged product?",
        intent=Intent(label="return_refund", confidence=0.90),
        entities=Entities(),
        retrieved_docs=[
            RetrievedDoc(
                content="Return/refund can be requested within 7 days after receiving the product.",
                source="return_policy",
                relevance=0.90
            )
        ],
    )
    print(f"User: How can I get a refund for my damaged product?")
    print(f"Bot: {response}")
    
    # Check memory
    print(f"\n📊 {get_conversation_summary()}")
    
    # Test connection
    print("\n🔌 Testing Groq Connection...")
    if test_groq_connection():
        print("✅ Groq API connected!")
    else:
        print("❌ Groq API connection failed.")
    
    print("\n" + "=" * 60)
    print("Testing complete!")