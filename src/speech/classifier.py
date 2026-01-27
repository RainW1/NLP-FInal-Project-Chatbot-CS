"""
Person A: Intent Classification Module
======================================

YOUR TASK:
1. Classify user message into intent categories
2. Return intent label with confidence score

INTENT CATEGORIES:
- complaint_delivery  : Keluhan pengiriman terlambat/tidak sampai
- complaint_product   : Keluhan produk rusak/salah/tidak sesuai
- order_status        : Cek status pesanan
- product_inquiry     : Tanya info produk, stok, harga
- return_refund       : Mau return/refund
- greeting            : Sapaan (halo, hi, selamat pagi)
- thanks              : Terima kasih
- other               : Lainnya

LIBRARIES TO USE:
- transformers (Hugging Face)
- torch

SETUP:
pip install transformers torch

MODEL OPTIONS:
1. Fine-tuned Indonesian BERT: "indobenchmark/indobert-base-p1"
2. Multilingual: "bert-base-multilingual-cased"
3. Or train your own on intent dataset!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from interfaces import Intent
from typing import List, Tuple

# ============================================================
# STUB IMPLEMENTATION - Replace with real code!
# ============================================================

def classify_intent(text: str) -> Intent:
    """
    STUB: Returns dummy intent based on keywords.
    
    TODO (Person A):
    1. Load fine-tuned classification model
    2. Preprocess text (tokenize, etc.)
    3. Run inference
    4. Return Intent with label and confidence
    
    Real implementation example:
    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("your-model")
    model = AutoModelForSequenceClassification.from_pretrained("your-model")
    
    LABELS = ["complaint_delivery", "complaint_product", "order_status", 
              "product_inquiry", "return_refund", "greeting", "thanks", "other"]
    
    def classify_intent(text: str) -> Intent:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
        confidence, predicted = torch.max(probs, dim=-1)
        label = LABELS[predicted.item()]
        
        return Intent(label=label, confidence=confidence.item())
    ```
    """
    # STUB - Simple keyword-based classification for testing
    text_lower = text.lower()
    
    # Keyword matching (replace with real model!)
    if any(word in text_lower for word in ["halo", "hi", "hey", "selamat"]):
        return Intent(label="greeting", confidence=0.95)
    
    elif any(word in text_lower for word in ["terima kasih", "thanks", "makasih"]):
        return Intent(label="thanks", confidence=0.90)
    
    elif any(word in text_lower for word in ["belum sampai", "terlambat", "lama", "tidak datang", "kemana"]):
        return Intent(label="complaint_delivery", confidence=0.88)
    
    elif any(word in text_lower for word in ["rusak", "cacat", "salah", "tidak sesuai", "beda"]):
        return Intent(label="complaint_product", confidence=0.85)
    
    elif any(word in text_lower for word in ["status", "dimana", "sampai mana", "tracking"]):
        return Intent(label="order_status", confidence=0.87)
    
    elif any(word in text_lower for word in ["refund", "return", "kembalikan", "balikin"]):
        return Intent(label="return_refund", confidence=0.89)
    
    elif any(word in text_lower for word in ["harga", "stok", "ada", "jual", "produk"]):
        return Intent(label="product_inquiry", confidence=0.82)
    
    else:
        return Intent(label="other", confidence=0.60)


def get_intent_description(intent_label: str) -> str:
    """
    Returns human-readable description of intent.
    Useful for debugging and UI display.
    """
    descriptions = {
        "complaint_delivery": "Keluhan Pengiriman",
        "complaint_product": "Keluhan Produk",
        "order_status": "Cek Status Pesanan",
        "product_inquiry": "Pertanyaan Produk",
        "return_refund": "Return/Refund",
        "greeting": "Sapaan",
        "thanks": "Terima Kasih",
        "other": "Lainnya"
    }
    return descriptions.get(intent_label, "Unknown")


# ============================================================
# TRAINING HELPER (Optional - if you want to fine-tune)
# ============================================================

def prepare_training_data() -> List[Tuple[str, str]]:
    """
    Sample training data for intent classification.
    
    TODO (Person A): Expand this dataset for better accuracy!
    
    Returns:
        List of (text, label) tuples
    """
    data = [
        # complaint_delivery
        ("pesanan saya belum sampai", "complaint_delivery"),
        ("sudah 5 hari barang tidak datang", "complaint_delivery"),
        ("pengiriman lama sekali", "complaint_delivery"),
        ("paket saya kemana ya", "complaint_delivery"),
        ("kok belum dikirim", "complaint_delivery"),
        
        # complaint_product
        ("barang yang datang rusak", "complaint_product"),
        ("produk tidak sesuai gambar", "complaint_product"),
        ("saya terima barang cacat", "complaint_product"),
        ("warna beda sama yang dipesan", "complaint_product"),
        
        # order_status
        ("status pesanan saya gimana", "order_status"),
        ("ORDER123 sudah sampai mana", "order_status"),
        ("tracking pesanan saya", "order_status"),
        ("cek resi dong", "order_status"),
        
        # product_inquiry
        ("harga iPhone berapa", "product_inquiry"),
        ("ada stok warna hitam", "product_inquiry"),
        ("produk ini ready tidak", "product_inquiry"),
        ("jual laptop gaming gak", "product_inquiry"),
        
        # return_refund
        ("saya mau refund", "return_refund"),
        ("bisa return barang ini", "return_refund"),
        ("mau kembalikan pesanan", "return_refund"),
        ("gimana cara refund", "return_refund"),
        
        # greeting
        ("halo", "greeting"),
        ("hi admin", "greeting"),
        ("selamat pagi", "greeting"),
        ("permisi", "greeting"),
        
        # thanks
        ("terima kasih", "thanks"),
        ("makasih banyak", "thanks"),
        ("thanks ya", "thanks"),
    ]
    return data


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test stub function
    test_messages = [
        "Halo, selamat pagi",
        "Pesanan ORDER123 saya belum sampai sudah seminggu",
        "Barang yang datang rusak, mau refund",
        "Status pesanan saya gimana ya?",
        "Ada stok iPhone 15 warna hitam?",
    ]
    
    print("Testing classify_intent stub:\n")
    for msg in test_messages:
        intent = classify_intent(msg)
        desc = get_intent_description(intent.label)
        print(f"Message: {msg}")
        print(f"Intent: {intent.label} ({desc})")
        print(f"Confidence: {intent.confidence:.2f}")
        print("-" * 50)
    
    print("\n✅ Stubs working! Ready for real implementation.")
