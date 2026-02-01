"""
Person C: Named Entity Recognition (NER) Module
===============================================

YOUR TASK:
1. Extract relevant entities from user messages
2. Focus on e-commerce specific entities:
   - Order ID (ORDER123, ORD-456789)
   - Product names
   - Dates/durations (kemarin, 5 hari, 2024-01-15)
   - Amounts (Rp 500.000)
   - Person names

LIBRARIES TO USE:
- transformers (for pre-trained NER models)
- spacy (alternative, good for Indonesian)
- regex (for pattern matching)

SETUP:
pip install transformers torch

MODEL OPTIONS:
1. HuggingFace NER: "dslim/bert-base-NER"
2. Indonesian NER: "cahya/bert-base-indonesian-NER"
3. Hybrid: Use model + regex patterns for specific formats
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from interfaces import Entities
from typing import List, Dict, Any

# ============================================================
# REGEX PATTERNS for E-commerce specific entities
# ============================================================

PATTERNS = {
    # Order ID patterns
    "order_id": [
        r'ORDER[-]?\d+',           # ORDER123, ORDER-123
        r'ORD[-]?\d+',             # ORD123, ORD-123
        r'#\d{6,}',                # #123456
        r'INV[-/]?\d+',            # INV123, INV/123
        r'[A-Z]{2,3}[-]?\d{8,}',   # JKT-12345678
    ],
    
    # Price/Amount patterns
    "amount": [
        r'Rp\.?\s*[\d.,]+',        # Rp 500.000, Rp.100000
        r'IDR\s*[\d.,]+',          # IDR 500000
        r'\d+\s*(?:ribu|juta|rb|jt)', # 500ribu, 1juta
    ],
    
    # Duration/Time patterns
    "duration": [
        r'\d+\s*(?:hari|minggu|bulan|tahun)',  # 5 hari, 2 minggu
        r'(?:kemarin|besok|lusa)',             # kemarin, besok
        r'\d+\s*(?:jam|menit)',                # 2 jam
        r'(?:tadi|barusan|baru saja)',         # tadi, barusan
    ],
    
    # Date patterns
    "date": [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # 15-01-2024, 15/1/24
        r'\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}',
        r'(?:senin|selasa|rabu|kamis|jumat|sabtu|minggu)\s+(?:lalu|kemarin|depan)',
    ],
}


# ============================================================
# STUB IMPLEMENTATION - Replace with real code!
# ============================================================

def extract_entities(text: str) -> Entities:
    """
    STUB: Extracts entities using regex patterns.
    
    TODO (Person C):
    1. Use pre-trained NER model for general entities
    2. Combine with regex for e-commerce specific patterns
    3. Post-process and validate extracted entities
    
    Real implementation example:
    ```python
    from transformers import pipeline
    
    # Load NER model
    ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    
    def extract_entities(text: str) -> Entities:
        entities = Entities()
        
        # Use regex for specific patterns (more reliable for IDs, amounts)
        entities.order_id = extract_with_regex(text, PATTERNS["order_id"])
        entities.amount = extract_with_regex(text, PATTERNS["amount"])
        entities.date = extract_with_regex(text, PATTERNS["duration"] + PATTERNS["date"])
        
        # Use NER model for general entities (names, products)
        ner_results = ner_model(text)
        for entity in ner_results:
            if entity['entity_group'] == 'PER':
                entities.person_name = entity['word']
            elif entity['entity_group'] == 'MISC':
                # Could be product name
                entities.product_name = entity['word']
        
        # Product name heuristics (common e-commerce products)
        entities.product_name = extract_product_name(text)
        
        return entities
    ```
    """
    entities = Entities()
    
    # Extract order ID
    entities.order_id = _extract_with_patterns(text, PATTERNS["order_id"])
    
    # Extract amount
    entities.amount = _extract_with_patterns(text, PATTERNS["amount"])
    
    # Extract duration/date
    date_patterns = PATTERNS["duration"] + PATTERNS["date"]
    entities.date = _extract_with_patterns(text, date_patterns)
    
    # Extract product name (simple heuristic - improve with NER model!)
    entities.product_name = _extract_product_name(text)
    
    # Extract person name (simple heuristic - improve with NER model!)
    entities.person_name = _extract_person_name(text)
    
    return entities


def _extract_with_patterns(text: str, patterns: List[str]) -> str:
    """
    Helper function to extract first match from list of regex patterns.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group().strip()
    return None


def _extract_product_name(text: str) -> str:
    """
    Simple product name extraction using common keywords.
    
    TODO (Person C): Improve this with NER model!
    """
    # Common product keywords (expand this list!)
    product_keywords = [
        # Electronics
        r'(?:iPhone|Samsung|Xiaomi|OPPO|Vivo|Realme)\s*\d*\s*(?:Pro|Plus|Max|Ultra)?',
        r'(?:MacBook|Laptop|HP|Dell|Asus|Lenovo)\s*(?:Pro|Air|Gaming)?',
        r'(?:iPad|Tablet|Tab)\s*(?:Pro|Air|Mini)?',
        r'(?:AirPods|Earbuds|TWS|Headphone|Headset)',
        r'(?:TV|Smart TV|LED|Monitor)\s*\d*\s*(?:inch)?',
        
        # Fashion
        r'(?:Sepatu|Tas|Baju|Celana|Kemeja|Kaos|Dress|Jaket)\s*(?:Nike|Adidas|Zara|H&M)?',
        
        # Generic patterns
        r'(?:produk|barang|pesanan)\s+([A-Za-z0-9\s]+)',
    ]
    
    for pattern in product_keywords:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group().strip()
    
    return None


def _extract_person_name(text: str) -> str:
    """
    Simple person name extraction.
    
    TODO (Person C): Improve this with NER model!
    """
    # Common patterns for names in customer service context
    name_patterns = [
        r'nama\s+(?:saya\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'saya\s+([A-Z][a-z]+)',
        r'(?:Bapak|Ibu|Pak|Bu|Mas|Mbak)\s+([A-Z][a-z]+)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None


def extract_all_entities_raw(text: str) -> Dict[str, List[str]]:
    """
    Extracts ALL matches for each entity type.
    Useful for debugging and analysis.
    
    Returns:
        Dict with entity type as key and list of all matches
    """
    results = {
        "order_ids": [],
        "amounts": [],
        "dates": [],
        "durations": []
    }
    
    for pattern in PATTERNS["order_id"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        results["order_ids"].extend(matches)
    
    for pattern in PATTERNS["amount"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        results["amounts"].extend(matches)
    
    for pattern in PATTERNS["date"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        results["dates"].extend(matches)
    
    for pattern in PATTERNS["duration"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        results["durations"].extend(matches)
    
    return results


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test messages
    test_messages = [
        "Pesanan ORDER123 saya belum sampai sudah 5 hari",
        "Halo, nama saya Budi. Mau komplain pesanan #789456 yang rusak",
        "iPhone 15 Pro Max yang saya pesan Rp 25.000.000 tidak sesuai",
        "Kemarin saya order laptop tapi sampai sekarang belum dikirim",
        "Tolong cek pesanan INV-20240115001 dong, sudah 2 minggu",
        "Saya Ibu Sari, mau refund sepatu Nike yang salah ukuran",
    ]
    
    print("Testing extract_entities stub:\n")
    print("=" * 70)
    
    for msg in test_messages:
        print(f"Message: {msg}")
        entities = extract_entities(msg)
        print(f"  Order ID    : {entities.order_id}")
        print(f"  Product     : {entities.product_name}")
        print(f"  Date        : {entities.date}")
        print(f"  Amount      : {entities.amount}")
        print(f"  Person Name : {entities.person_name}")
        print("-" * 70)
    
    print("\n✅ Stubs working! Ready for real implementation.")
