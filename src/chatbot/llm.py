"""
Person B: Hybrid Chatbot Module (Rules + LLM)
=============================================

COMPLETE IMPLEMENTATION with:
- Groq API (Llama3-8B) - Pre-trained
- Flan-T5 - Fine-tuned (local)
- Model switching for comparison
- Multilingual support (auto-detect language)
- Conversation memory
- Rule-based for simple intents

SETUP:
pip install groq transformers torch

Get Groq API key: https://console.groq.com
"""

import os
import random
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

# ============================================================
# IMPORTS
# ============================================================

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Please install: pip install groq")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Please install: pip install transformers torch")


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

# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Model options
MODELS = {
    "pretrained": {
        "name": "llama-3.1-8b-instant",
        "type": "groq",
        "description": "Llama 3.1 8B via Groq API (Pre-trained)"
    },
    "finetuned": {
        "name": "google/flan-t5-base",
        "type": "local",
        "description": "Flan-T5 Base (Fine-tuned on instructions)"
    }
}

# Default model
CURRENT_MODEL = "pretrained"

# Response settings
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_HISTORY_LENGTH = 20


# ============================================================
# MODEL SWITCHING
# ============================================================

def set_model(model_type: str) -> str:
    """
    Switch between models
    
    Args:
        model_type: "pretrained" or "finetuned"
    
    Returns:
        Confirmation message
    """
    global CURRENT_MODEL
    
    if model_type not in MODELS:
        return f"❌ Invalid model. Choose: {list(MODELS.keys())}"
    
    CURRENT_MODEL = model_type
    model_info = MODELS[model_type]
    return f"✅ Switched to {model_type}: {model_info['description']}"


def get_current_model() -> Dict:
    """Get info about current model"""
    return {
        "type": CURRENT_MODEL,
        **MODELS[CURRENT_MODEL]
    }


def list_models() -> Dict:
    """List all available models"""
    return MODELS


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """Manages conversation history for context"""
    
    def __init__(self, max_length: int = MAX_HISTORY_LENGTH):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
    
    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]
    
    def get_history(self, last_n: int = 10) -> List[Dict[str, str]]:
        return self.history[-last_n:]
    
    def clear(self):
        self.history = []
    
    def __len__(self):
        return len(self.history)


_memory = ConversationMemory()

def get_memory() -> ConversationMemory:
    return _memory

def clear_memory():
    _memory.clear()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:
    """Detect language: 'id' (Indonesian), 'en' (English), or 'other'"""
    text_lower = text.lower()
    
    id_words = [
        "aku", "saya", "mau", "bisa", "tolong", "terima kasih", "halo",
        "gimana", "bagaimana", "apa", "dimana", "kapan", "sudah", "belum",
        "dong", "ya", "gak", "tidak", "ini", "itu", "kamu", "anda",
        "pesanan", "barang", "kirim", "sampai", "lama", "refund", "komplain"
    ]
    
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
    return "other"


# ============================================================
# RULE-BASED RESPONSES (Multilingual)
# ============================================================

RULE_BASED_RESPONSES = {
    "greeting": {
        "id": [
            "Halo! 👋 Selamat datang di Customer Service kami. Ada yang bisa saya bantu?",
            "Hi! Terima kasih sudah menghubungi kami. Silakan sampaikan pertanyaan Anda 😊",
        ],
        "en": [
            "Hello! 👋 Welcome to our Customer Service. How can I help you today?",
            "Hi! Thank you for contacting us. How can I assist you? 😊",
        ],
        "other": ["Hello! 👋 How can I help you today?"]
    },
    "thanks": {
        "id": [
            "Sama-sama! 😊 Senang bisa membantu. Ada lagi yang bisa saya bantu?",
            "Terima kasih kembali! Jangan ragu hubungi kami lagi ya.",
        ],
        "en": [
            "You're welcome! 😊 Happy to help. Anything else I can assist with?",
            "No problem! Feel free to reach out again if you need help.",
        ],
        "other": ["You're welcome! 😊 Anything else I can help with?"]
    },
    "goodbye": {
        "id": ["Terima kasih sudah menghubungi kami. Sampai jumpa! 👋"],
        "en": ["Thank you for contacting us. Goodbye! Have a great day! 👋"],
        "other": ["Goodbye! Have a great day! 👋"]
    },
}


def get_rule_based_response(intent: Intent, language: str) -> Optional[str]:
    """Get rule-based response for simple intents"""
    if intent.label in RULE_BASED_RESPONSES and intent.confidence > 0.85:
        responses = RULE_BASED_RESPONSES[intent.label]
        lang_responses = responses.get(language, responses.get("other", responses.get("en")))
        return random.choice(lang_responses)
    return None


# ============================================================
# MODEL A: PRE-TRAINED (Groq/Llama)
# ============================================================

def create_groq_client() -> Optional[Groq]:
    """Create Groq client"""
    if not GROQ_AVAILABLE or GROQ_API_KEY == "YOUR_API_KEY_HERE":
        return None
    return Groq(api_key=GROQ_API_KEY)


def generate_groq_response(
    user_message: str,
    system_prompt: str,
    memory: ConversationMemory
) -> Tuple[str, float]:
    """Generate response using Groq/Llama (pre-trained)"""
    
    client = create_groq_client()
    if client is None:
        return "Groq API not configured", 0
    
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in memory.get_history(last_n=10):
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=MODELS["pretrained"]["name"],
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        latency = (time.time() - start_time) * 1000
        return response.choices[0].message.content.strip(), latency
    
    except Exception as e:
        return f"Error: {e}", 0


# ============================================================
# MODEL B: FINE-TUNED (Flan-T5)
# ============================================================

_finetuned_model = None
_finetuned_tokenizer = None

def load_finetuned_model():
    """Load fine-tuned model (singleton)"""
    global _finetuned_model, _finetuned_tokenizer
    
    if _finetuned_model is None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        print(f"🔄 Loading fine-tuned model: {MODELS['finetuned']['name']}...")
        _finetuned_tokenizer = AutoTokenizer.from_pretrained(MODELS["finetuned"]["name"])
        _finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(MODELS["finetuned"]["name"])
        print("✅ Fine-tuned model loaded!")
    
    return _finetuned_model, _finetuned_tokenizer


def generate_finetuned_response(
    user_message: str,
    context: str,
    memory: ConversationMemory
) -> Tuple[str, float]:
    """Generate response using Flan-T5 (fine-tuned)"""
    
    if not TRANSFORMERS_AVAILABLE:
        return "Transformers not available", 0
    
    try:
        model, tokenizer = load_finetuned_model()
        
        # Build conversation context
        history_text = ""
        for msg in memory.get_history(last_n=4):
            role = "Customer" if msg["role"] == "user" else "Agent"
            history_text += f"{role}: {msg['content']}\n"
        
        # Format input for Flan-T5
        input_text = f"""You are a helpful customer service agent. Answer the customer's question based on the context.

Context information:
{context}

Conversation history:
{history_text}

Customer: {user_message}

Agent:"""

        start_time = time.time()
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True,
            do_sample=True,
            temperature=0.7
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = (time.time() - start_time) * 1000
        
        return response, latency
    
    except Exception as e:
        return f"Error: {e}", 0


# ============================================================
# SYSTEM PROMPT BUILDER
# ============================================================

def build_system_prompt(
    intent: Intent,
    entities: Entities,
    retrieved_docs: List[RetrievedDoc],
    language: str
) -> str:
    """Build system prompt with RAG context"""
    
    # Context from RAG
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(f"[{doc.source}]: {doc.content}")
    context = "\n".join(context_parts) if context_parts else "No specific information available."
    
    # Entity info
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
LANGUAGE INSTRUCTION:
Always respond in the SAME LANGUAGE as the user's message!
Detected language: {language}
═══════════════════════════════════════════════════════════════

CONVERSATION MEMORY:
- You have memory of this conversation
- Reference previous context naturally
- NEVER ask for information the user already provided

KNOWLEDGE BASE:
{context}

DETECTED ENTITIES:
{entity_text}

INTENT: {intent.label} (confidence: {intent.confidence:.0%})

GUIDELINES:
- Be friendly, warm, and professional
- Keep responses concise (2-4 sentences)
- Use 1-2 emojis appropriately
- Provide clear, actionable solutions
- If order number exists, mention it for confirmation"""

    return system_prompt


# ============================================================
# MAIN FUNCTION (matches interface contract)
# ============================================================

def generate_response(
    user_message: str,
    intent: Intent,
    entities: Entities,
    retrieved_docs: List[RetrievedDoc],
    conversation_history: List[Message] = None,
    model_type: str = None  # "pretrained" or "finetuned" (None = use current)
) -> str:
    """
    Generates chatbot response using selected model.
    
    Features:
    - Model switching (pretrained vs finetuned)
    - Multilingual (auto-detects and responds in same language)
    - Conversation memory
    - Rule-based for simple intents
    - RAG context integration
    
    Args:
        user_message: Current user message
        intent: Classified intent
        entities: Extracted entities
        retrieved_docs: Relevant docs from RAG
        conversation_history: Previous messages (optional)
        model_type: "pretrained" or "finetuned" (optional, uses current if None)
        
    Returns:
        Bot response as string
    """
    
    # Use specified model or current default
    active_model = model_type if model_type else CURRENT_MODEL
    
    # Get memory
    memory = get_memory()
    
    # Detect language
    language = detect_language(user_message)
    
    # Try rule-based response for simple intents
    rule_response = get_rule_based_response(intent, language)
    if rule_response:
        memory.add("user", user_message)
        memory.add("assistant", rule_response)
        return rule_response
    
    # Build context
    system_prompt = build_system_prompt(intent, entities, retrieved_docs, language)
    context = "\n".join([doc.content for doc in retrieved_docs])
    
    # Generate response based on selected model
    if active_model == "pretrained":
        response, latency = generate_groq_response(user_message, system_prompt, memory)
    else:
        response, latency = generate_finetuned_response(user_message, context, memory)
    
    # Save to memory
    memory.add("user", user_message)
    memory.add("assistant", response)
    
    return response


def generate_response_with_comparison(
    user_message: str,
    intent: Intent,
    entities: Entities,
    retrieved_docs: List[RetrievedDoc],
) -> Dict:
    """
    Generate responses from BOTH models for comparison.
    Useful for demo/presentation.
    
    Returns:
        Dict with both responses and latencies
    """
    
    memory = get_memory()
    language = detect_language(user_message)
    
    # Check rule-based first
    rule_response = get_rule_based_response(intent, language)
    if rule_response:
        return {
            "pretrained": {"response": rule_response, "latency_ms": 0, "note": "Rule-based"},
            "finetuned": {"response": rule_response, "latency_ms": 0, "note": "Rule-based"},
            "is_rule_based": True
        }
    
    # Build prompts
    system_prompt = build_system_prompt(intent, entities, retrieved_docs, language)
    context = "\n".join([doc.content for doc in retrieved_docs])
    
    # Generate from both models
    pretrained_resp, pretrained_lat = generate_groq_response(user_message, system_prompt, memory)
    finetuned_resp, finetuned_lat = generate_finetuned_response(user_message, context, memory)
    
    return {
        "pretrained": {
            "response": pretrained_resp,
            "latency_ms": round(pretrained_lat, 2),
            "model": MODELS["pretrained"]["name"]
        },
        "finetuned": {
            "response": finetuned_resp,
            "latency_ms": round(finetuned_lat, 2),
            "model": MODELS["finetuned"]["name"]
        },
        "is_rule_based": False
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def test_groq_connection() -> bool:
    """Test Groq API connection"""
    client = create_groq_client()
    if client is None:
        return False
    
    try:
        response = client.chat.completions.create(
            model=MODELS["pretrained"]["name"],
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        return True
    except:
        return False


def test_finetuned_model() -> bool:
    """Test fine-tuned model loading"""
    try:
        load_finetuned_model()
        return True
    except:
        return False


def get_model_status() -> Dict:
    """Get status of all models"""
    return {
        "current_model": CURRENT_MODEL,
        "pretrained_available": test_groq_connection(),
        "finetuned_available": TRANSFORMERS_AVAILABLE,
        "models": MODELS
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Hybrid Chatbot with Model Switching")
    print("=" * 70)
    
    # Clear memory
    clear_memory()
    
    # Show available models
    print("\n📋 Available Models:")
    for key, info in MODELS.items():
        print(f"   {key}: {info['description']}")
    
    print(f"\n🎯 Current model: {CURRENT_MODEL}")
    
    # Test case
    test_message = "My order ORDER123 hasn't arrived yet. It's been 5 days."
    test_intent = Intent(label="complaint_delivery", confidence=0.88)
    test_entities = Entities(order_id="ORDER123", date="5 days")
    test_docs = [
        RetrievedDoc(
            content="Standard shipping takes 3-5 business days for domestic orders.",
            source="shipping_policy",
            relevance=0.85
        ),
        RetrievedDoc(
            content="If your order hasn't arrived, please contact customer service with your order number.",
            source="faq",
            relevance=0.80
        )
    ]
    
    # Test with both models
    print("\n" + "=" * 70)
    print("📝 Test: " + test_message)
    print("=" * 70)
    
    # Pre-trained
    print("\n🅰️ PRE-TRAINED (Llama-3.1-8B):")
    set_model("pretrained")
    response = generate_response(test_message, test_intent, test_entities, test_docs)
    print(f"   {response}")
    
    # Fine-tuned
    print("\n🅱️ FINE-TUNED (Flan-T5):")
    set_model("finetuned")
    clear_memory()  # Clear memory for fair comparison
    response = generate_response(test_message, test_intent, test_entities, test_docs)
    print(f"   {response}")
    
    # Test comparison function
    print("\n" + "=" * 70)
    print("📊 Side-by-side Comparison:")
    print("=" * 70)
    
    clear_memory()
    comparison = generate_response_with_comparison(
        test_message, test_intent, test_entities, test_docs
    )
    
    print(f"\n🅰️ Pre-trained ({comparison['pretrained']['latency_ms']}ms):")
    print(f"   {comparison['pretrained']['response']}")
    
    print(f"\n🅱️ Fine-tuned ({comparison['finetuned']['latency_ms']}ms):")
    print(f"   {comparison['finetuned']['response']}")
    
    # Model status
    print("\n" + "=" * 70)
    print("📊 Model Status:")
    status = get_model_status()
    print(f"   Current: {status['current_model']}")
    print(f"   Groq API: {'✅' if status['pretrained_available'] else '❌'}")
    print(f"   Transformers: {'✅' if status['finetuned_available'] else '❌'}")
    
    print("\n✅ Testing complete!")