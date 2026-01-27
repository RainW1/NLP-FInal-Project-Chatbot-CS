"""
Person B: Hybrid Chatbot Module (Rules + LLM)
=============================================

YOUR TASK:
1. Implement rule-based responses for simple intents (greeting, FAQ)
2. Implement LLM-based responses for complex queries
3. Use RAG context to enhance responses

LIBRARIES TO USE:
- groq (for Llama3/Mixtral - FAST & FREE)
- OR google-generativeai (for Gemini - FREE)

SETUP:
pip install groq google-generativeai

API KEYS:
- Groq: https://console.groq.com (free, fast!)
- Google AI Studio: https://makersuite.google.com/app/apikey (free)

HYBRID APPROACH:
- Simple intents (greeting, thanks) → Rule-based (instant, no API)
- Complex queries → LLM with RAG context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from interfaces import Intent, Entities, RetrievedDoc, Message
from typing import List, Optional

# ============================================================
# CONFIGURATION
# ============================================================

# Load from environment variable in production!
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key-here")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-api-key-here")

# Choose your LLM provider
LLM_PROVIDER = "groq"  # Options: "groq", "google", "stub"


# ============================================================
# RULE-BASED RESPONSES (for simple intents)
# ============================================================

RULE_BASED_RESPONSES = {
    "greeting": [
        "Halo! Selamat datang di Customer Service kami. Ada yang bisa saya bantu? 😊",
        "Hi! Terima kasih sudah menghubungi kami. Apa yang bisa saya bantu hari ini?",
        "Selamat datang! Saya siap membantu Anda. Silakan sampaikan pertanyaan atau keluhan Anda.",
    ],
    "thanks": [
        "Sama-sama! Senang bisa membantu. Ada lagi yang bisa saya bantu?",
        "Terima kasih kembali! Jangan ragu untuk menghubungi kami lagi jika ada pertanyaan.",
        "You're welcome! Semoga harinya menyenangkan 😊",
    ],
}


def get_rule_based_response(intent: Intent) -> Optional[str]:
    """
    Returns rule-based response if available for the intent.
    Returns None if LLM should handle it.
    """
    import random
    
    if intent.label in RULE_BASED_RESPONSES and intent.confidence > 0.85:
        return random.choice(RULE_BASED_RESPONSES[intent.label])
    
    return None


# ============================================================
# STUB IMPLEMENTATION - Replace with real code!
# ============================================================

def generate_response(
    user_message: str,
    intent: Intent,
    entities: Entities,
    retrieved_docs: List[RetrievedDoc],
    conversation_history: List[Message]
) -> str:
    """
    STUB: Generates response using hybrid approach.
    
    TODO (Person B):
    1. Check if rule-based response is available
    2. If not, use LLM with RAG context
    3. Personalize response with extracted entities
    
    Real implementation with Groq:
    ```python
    from groq import Groq
    
    client = Groq(api_key=GROQ_API_KEY)
    
    def generate_response(...) -> str:
        # Try rule-based first
        rule_response = get_rule_based_response(intent)
        if rule_response:
            return rule_response
        
        # Build context from RAG
        context = "\\n".join([doc.content for doc in retrieved_docs])
        
        # Build conversation history
        messages = [
            {"role": "system", "content": f'''Kamu adalah customer service e-commerce yang ramah dan helpful.
            Gunakan informasi berikut untuk menjawab:
            {context}
            
            Entity yang terdeteksi:
            - Order ID: {entities.order_id}
            - Produk: {entities.product_name}
            - Tanggal: {entities.date}
            
            Intent pelanggan: {intent.label}
            
            Jawab dengan ramah, singkat, dan helpful dalam Bahasa Indonesia.'''}
        ]
        
        # Add conversation history
        for msg in conversation_history[-5:]:  # Last 5 messages
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Call LLM
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    ```
    """
    # Try rule-based first
    rule_response = get_rule_based_response(intent)
    if rule_response:
        return rule_response
    
    # STUB - Template-based response for testing
    # Replace this with real LLM call!
    
    # Build personalized response based on intent and entities
    response_templates = {
        "complaint_delivery": f"""Mohon maaf atas ketidaknyamanan yang Anda alami{f' dengan pesanan {entities.order_id}' if entities.order_id else ''}.
        
Kami memahami pesanan Anda{f' sudah {entities.date}' if entities.date else ''} belum sampai. Tim kami akan segera mengecek status pengiriman dan menghubungi pihak ekspedisi.

Mohon tunggu 1x24 jam untuk update status. Apakah ada yang lain yang bisa saya bantu?""",

        "complaint_product": f"""Mohon maaf atas ketidaknyamanan yang Anda alami{f' dengan {entities.product_name}' if entities.product_name else ''}.

Kami sangat menyesal mendengar produk tidak sesuai harapan. Silakan kirimkan foto bukti melalui aplikasi, dan tim kami akan segera memproses penggantian atau refund.

Apakah Anda ingin saya bantu proses komplainnya sekarang?""",

        "order_status": f"""Baik, saya akan cek status pesanan{f' {entities.order_id}' if entities.order_id else ' Anda'}.

Untuk mengecek status pesanan, Anda juga bisa:
1. Buka menu "Pesanan Saya" di aplikasi
2. Masukkan nomor resi di halaman tracking

Mohon tunggu sebentar ya, saya cek dulu sistemnya.""",

        "return_refund": f"""Baik, saya akan bantu proses return/refund{f' untuk {entities.product_name}' if entities.product_name else ''}.

Syarat return/refund:
• Diajukan dalam 7 hari setelah barang diterima
• Barang dalam kondisi original
• Proses refund 3-5 hari kerja setelah barang kami terima

Apakah Anda ingin melanjutkan proses return?""",

        "product_inquiry": f"""Terima kasih atas pertanyaannya{f' tentang {entities.product_name}' if entities.product_name else ''}.

Untuk informasi produk terlengkap, silakan cek halaman produk di aplikasi atau website kami. 

Apakah ada spesifikasi tertentu yang ingin Anda tanyakan?""",

        "other": """Terima kasih sudah menghubungi kami.

Mohon maaf, saya kurang memahami pertanyaan Anda. Bisa tolong jelaskan lebih detail apa yang bisa saya bantu?

Atau Anda bisa langsung menghubungi hotline kami di 021-12345678 untuk bantuan lebih lanjut."""
    }
    
    return response_templates.get(intent.label, response_templates["other"])


def generate_response_with_groq(
    user_message: str,
    system_prompt: str,
    conversation_history: List[Message]
) -> str:
    """
    Generate response using Groq API (Llama3).
    
    TODO (Person B): Implement real Groq API call
    """
    # STUB - Return dummy response
    return f"[GROQ Response to: {user_message}]"


def generate_response_with_google(
    user_message: str,
    system_prompt: str,
    conversation_history: List[Message]
) -> str:
    """
    Generate response using Google Gemini API.
    
    TODO (Person B): Implement real Gemini API call
    """
    # STUB - Return dummy response
    return f"[GEMINI Response to: {user_message}]"


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    from interfaces import Intent, Entities, RetrievedDoc, Message
    
    print("Testing generate_response stub:\n")
    
    # Test case 1: Greeting (should use rule-based)
    print("Test 1: Greeting")
    response = generate_response(
        user_message="Halo selamat pagi",
        intent=Intent(label="greeting", confidence=0.95),
        entities=Entities(),
        retrieved_docs=[],
        conversation_history=[]
    )
    print(f"Response: {response}\n")
    print("-" * 60)
    
    # Test case 2: Delivery complaint
    print("Test 2: Delivery Complaint")
    response = generate_response(
        user_message="Pesanan ORDER123 belum sampai sudah 5 hari",
        intent=Intent(label="complaint_delivery", confidence=0.88),
        entities=Entities(order_id="ORDER123", date="5 hari"),
        retrieved_docs=[
            RetrievedDoc(
                content="Pengiriman standar 3-5 hari kerja",
                source="shipping_policy.txt",
                relevance=0.85
            )
        ],
        conversation_history=[]
    )
    print(f"Response: {response}\n")
    print("-" * 60)
    
    # Test case 3: Return/Refund
    print("Test 3: Return Request")
    response = generate_response(
        user_message="Saya mau refund iPhone yang rusak",
        intent=Intent(label="return_refund", confidence=0.90),
        entities=Entities(product_name="iPhone"),
        retrieved_docs=[],
        conversation_history=[]
    )
    print(f"Response: {response}\n")
    
    print("\n✅ Stubs working! Ready for real implementation.")
