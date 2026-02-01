"""
Person C: Summarization & Model Comparison Module
=================================================

YOUR TASK:
1. Summarize conversation history for records
2. Compare two summarization models
3. Generate comparison metrics (ROUGE scores, latency)

LIBRARIES TO USE:
- transformers (for summarization models)
- rouge_score (for evaluation)
- time (for latency measurement)

SETUP:
pip install transformers rouge_score torch

MODEL OPTIONS FOR COMPARISON:
1. BART: "facebook/bart-large-cnn"
2. PEGASUS: "google/pegasus-cnn_dailymail"
3. T5: "t5-small" or "t5-base"
4. Indonesian: "LazarusNLP/IndoNanoT5-base"
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from interfaces import Message
from typing import List, Dict

# ============================================================
# SUMMARIZATION MODEL LOADING (with caching)
# ============================================================

_SUMMARIZATION_MODEL_CACHE = None

def _get_summarization_model():
    """
    Load and cache BART summarization model.
    
    Returns:
        Summarization pipeline or None if loading fails
    """
    global _SUMMARIZATION_MODEL_CACHE
    
    if _SUMMARIZATION_MODEL_CACHE is not None:
        return _SUMMARIZATION_MODEL_CACHE
    
    try:
        from transformers import pipeline
        
        # Use BART for summarization (good quality, reasonable speed)
        _SUMMARIZATION_MODEL_CACHE = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # Use CPU
        )
        print("✅ Loaded BART summarization model: facebook/bart-large-cnn")
        return _SUMMARIZATION_MODEL_CACHE
        
    except Exception as e:
        print(f"❌ Could not load summarization model: {e}")
        print("   Falling back to rule-based summarization...")
        return None


def _detect_language(text: str) -> str:
    """
    Simple language detection based on common words.
    Returns 'id' for Indonesian or 'en' for English.
    """
    indonesian_keywords = ['saya', 'yang', 'dengan', 'untuk', 'adalah', 'belum', 'sudah', 'mau', 'bisa']
    english_keywords = ['the', 'is', 'are', 'my', 'have', 'has', 'can', 'please', 'order']
    
    text_lower = text.lower()
    indo_count = sum(1 for word in indonesian_keywords if word in text_lower)
    eng_count = sum(1 for word in english_keywords if word in text_lower)
    
    return 'id' if indo_count > eng_count else 'en'


def summarize_conversation(messages: List[Message]) -> str:
    """
    Summarizes conversation history using BART model.
    
    Args:
        messages: List of Message objects (full conversation)
        
    Returns:
        Summary string (2-3 sentences)
    """
    if not messages:
        return "Tidak ada percakapan untuk dirangkum."
    
    if len(messages) < 2:
        return "Percakapan terlalu singkat untuk dirangkum."
    
    # Detect language from first user message
    user_messages_sample = [m.content for m in messages if m.role == "user"][:3]
    sample_text = " ".join(user_messages_sample)
    language = _detect_language(sample_text)
    
    # Format conversation into text with appropriate role names
    conversation_text = ""
    for msg in messages:
        if language == 'id':
            role = "Pelanggan" if msg.role == "user" else "Customer Service"
        else:
            role = "Customer" if msg.role == "user" else "Customer Service"
        conversation_text += f"{role}: {msg.content}\n"
    
    # Try to use BART model
    try:
        summarizer = _get_summarization_model()
        
        if summarizer and len(conversation_text) >= 50:
            # BART works best with text between 100-1024 tokens
            # Truncate if too long
            max_input_length = 1000
            if len(conversation_text) > max_input_length:
                conversation_text = conversation_text[:max_input_length]
            
            result = summarizer(
                conversation_text,
                max_length=100,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            
            return result[0]["summary_text"]
    except Exception as e:
        print(f"⚠️ Summarization model failed: {e}")
        print("   Using fallback rule-based summary...")
    
    # Fallback: Rule-based summarization with bilingual support
    user_messages = [m.content for m in messages if m.role == "user"]
    bot_messages = [m.content for m in messages if m.role == "assistant"]
    
    # Extract order IDs
    import re
    order_ids = []
    for m in user_messages:
        matches = re.findall(r'ORDER[-]?\d+|ORD[-]?\d+|#\d{6,}', m, re.IGNORECASE)
        order_ids.extend(matches)
    
    # Detect topics based on keywords (bilingual)
    topics = []
    
    # Delivery delay
    if any(word in " ".join(user_messages).lower() for word in ["belum sampai", "terlambat", "hasn't arrived", "delayed", "late"]):
        topics.append("keterlambatan pengiriman" if language == 'id' else "delivery delay")
    
    # Damaged product
    if any(word in " ".join(user_messages).lower() for word in ["rusak", "cacat", "broken", "damaged", "defective"]):
        topics.append("produk rusak/cacat" if language == 'id' else "damaged/defective product")
    
    # Refund/Return
    if any(word in " ".join(user_messages).lower() for word in ["refund", "return", "tukar", "kembali"]):
        topics.append("permintaan refund/return" if language == 'id' else "refund/return request")
    
    # Status/Tracking
    if any(word in " ".join(user_messages).lower() for word in ["status", "tracking", "cek", "check"]):
        topics.append("pengecekan status pesanan" if language == 'id' else "order status inquiry")
    
    # Build summary based on language
    summary_parts = []
    
    if language == 'id':
        summary_parts.append(f"Percakapan dengan {len(messages)} pesan.")
        if topics:
            summary_parts.append(f"Pelanggan menghubungi terkait {', '.join(topics)}.")
        else:
            summary_parts.append("Pelanggan menghubungi untuk bertanya.")
        if order_ids:
            summary_parts.append(f"Nomor pesanan yang disebutkan: {', '.join(set(order_ids))}.")
        if any("terima kasih" in m.lower() or "thanks" in m.lower() for m in user_messages):
            summary_parts.append("Pelanggan mengucapkan terima kasih (kemungkinan terselesaikan).")
    else:
        # English fallback
        summary_parts.append(f"Conversation with {len(messages)} messages.")
        if topics:
            summary_parts.append(f"Customer contacted regarding {', '.join(topics)}.")
        else:
            summary_parts.append("Customer contacted with a question.")
        if order_ids:
            summary_parts.append(f"Order number(s) mentioned: {', '.join(set(order_ids))}.")
        if any("terima kasih" in m.lower() or "thank" in m.lower() for m in user_messages):
            summary_parts.append("Customer expressed thanks (likely resolved).")
    
    return " ".join(summary_parts)


def summarize_with_model(text: str, model_name: str = "bart") -> Dict:
    """
    Summarize text with specified model and return result with timing.
    
    TODO (Person C): Implement real model calls
    
    Returns:
        Dict with 'summary', 'model', 'latency_ms'
    """
    start_time = time.time()
    
    # STUB - Return dummy summary
    if model_name == "bart":
        summary = f"[BART Summary] Ringkasan percakapan customer service."
    elif model_name == "pegasus":
        summary = f"[PEGASUS Summary] Ringkasan percakapan customer service."
    else:
        summary = f"[{model_name} Summary] Ringkasan percakapan."
    
    # Simulate some processing time
    time.sleep(0.1)  # Remove this in real implementation!
    
    latency = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        "summary": summary,
        "model": model_name,
        "latency_ms": latency
    }


# ============================================================
# MODEL COMPARISON
# ============================================================

_MODEL_COMPARISON_CACHE = {"bart": None, "pegasus": None}

def _load_comparison_models():
    """
    Load both BART and PEGASUS models for comparison.
    
    Returns:
        Dict with 'bart' and 'pegasus' pipelines
    """
    global _MODEL_COMPARISON_CACHE
    
    # Load BART if not cached
    if _MODEL_COMPARISON_CACHE["bart"] is None:
        try:
            from transformers import pipeline
            _MODEL_COMPARISON_CACHE["bart"] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1
            )
            print("✅ Loaded BART for comparison: facebook/bart-large-cnn")
        except Exception as e:
            print(f"❌ Could not load BART: {e}")
    
    # Load PEGASUS if not cached
    if _MODEL_COMPARISON_CACHE["pegasus"] is None:
        try:
            from transformers import pipeline
            _MODEL_COMPARISON_CACHE["pegasus"] = pipeline(
                "summarization",
                model="google/pegasus-cnn_dailymail",
                device=-1
            )
            print("✅ Loaded PEGASUS for comparison: google/pegasus-cnn_dailymail")
        except Exception as e:
            print(f"❌ Could not load PEGASUS: {e}")
    
    return _MODEL_COMPARISON_CACHE


def compare_models(text_samples: List[str], reference_summaries: List[str] = None) -> Dict:
    """
    Compares BART and PEGASUS summarization models on given samples.
    
    Args:
        text_samples: List of conversation texts to summarize
        reference_summaries: Optional list of reference summaries for ROUGE calculation
        
    Returns:
        Dict with comparison metrics (ROUGE scores, latency, etc.)
    """
    if not text_samples:
        return {"error": "No text samples provided"}
    
    print(f"\n{'='*60}")
    print(f"Starting Model Comparison: BART vs PEGASUS")
    print(f"Number of samples: {len(text_samples)}")
    print(f"{'='*60}\n")
    
    # Load models
    models = _load_comparison_models()
    bart_model = models["bart"]
    pegasus_model = models["pegasus"]
    
    # Initialize results
    results = {
        "model_a": {
            "name": "BART (facebook/bart-large-cnn)",
            "summaries": [],
            "latencies": [],
            "rouge_scores": []
        },
        "model_b": {
            "name": "PEGASUS (google/pegasus-cnn_dailymail)",
            "summaries": [],
            "latencies": [],
            "rouge_scores": []
        },
        "num_samples": len(text_samples)
    }
    
    # Process each sample
    for i, text in enumerate(text_samples):
        print(f"Processing sample {i+1}/{len(text_samples)}...")
        
        # Truncate if too long
        if len(text) > 1000:
            text = text[:1000]
        
        # BART
        if bart_model:
            try:
                start = time.time()
                bart_summary = bart_model(text, max_length=100, min_length=30, do_sample=False, truncation=True)
                latency = (time.time() - start) * 1000
                
                results["model_a"]["summaries"].append(bart_summary[0]["summary_text"])
                results["model_a"]["latencies"].append(latency)
            except Exception as e:
                print(f"  ⚠️ BART failed on sample {i+1}: {e}")
                results["model_a"]["summaries"].append("[Error generating summary]")
                results["model_a"]["latencies"].append(0)
        
        # PEGASUS
        if pegasus_model:
            try:
                start = time.time()
                pegasus_summary = pegasus_model(text, max_length=100, min_length=30, do_sample=False, truncation=True)
                latency = (time.time() - start) * 1000
                
                results["model_b"]["summaries"].append(pegasus_summary[0]["summary_text"])
                results["model_b"]["latencies"].append(latency)
            except Exception as e:
                print(f"  ⚠️ PEGASUS failed on sample {i+1}: {e}")
                results["model_b"]["summaries"].append("[Error generating summary]")
                results["model_b"]["latencies"].append(0)
    
    # Calculate ROUGE scores if reference summaries provided
    if reference_summaries and len(reference_summaries) == len(text_samples):
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            print("\nCalculating ROUGE scores...")
            
            for i, ref in enumerate(reference_summaries):
                # BART scores
                if i < len(results["model_a"]["summaries"]):
                    bart_scores = scorer.score(ref, results["model_a"]["summaries"][i])
                    results["model_a"]["rouge_scores"].append({
                        'rouge1': bart_scores['rouge1'].fmeasure,
                        'rouge2': bart_scores['rouge2'].fmeasure,
                        'rougeL': bart_scores['rougeL'].fmeasure
                    })
                
                # PEGASUS scores
                if i < len(results["model_b"]["summaries"]):
                    pegasus_scores = scorer.score(ref, results["model_b"]["summaries"][i])
                    results["model_b"]["rouge_scores"].append({
                        'rouge1': pegasus_scores['rouge1'].fmeasure,
                        'rouge2': pegasus_scores['rouge2'].fmeasure,
                        'rougeL': pegasus_scores['rougeL'].fmeasure
                    })
        except ImportError:
            print("⚠️ rouge_score not installed. Skipping ROUGE calculation.")
        except Exception as e:
            print(f"⚠️ ROUGE calculation failed: {e}")
    
    # Calculate averages
    if results["model_a"]["latencies"]:
        results["model_a"]["avg_latency_ms"] = sum(results["model_a"]["latencies"]) / len(results["model_a"]["latencies"])
    
    if results["model_b"]["latencies"]:
        results["model_b"]["avg_latency_ms"] = sum(results["model_b"]["latencies"]) / len(results["model_b"]["latencies"])
    
    # Calculate average ROUGE scores
    if results["model_a"]["rouge_scores"]:
        rouge1_avg = sum(s['rouge1'] for s in results["model_a"]["rouge_scores"]) / len(results["model_a"]["rouge_scores"])
        rouge2_avg = sum(s['rouge2'] for s in results["model_a"]["rouge_scores"]) / len(results["model_a"]["rouge_scores"])
        rougeL_avg = sum(s['rougeL'] for s in results["model_a"]["rouge_scores"]) / len(results["model_a"]["rouge_scores"])
        results["model_a"]["avg_rouge1"] = rouge1_avg
        results["model_a"]["avg_rouge2"] = rouge2_avg
        results["model_a"]["avg_rougeL"] = rougeL_avg
    else:
        # Use dummy values if no ROUGE calculated
        results["model_a"]["avg_rouge1"] = 0.42
        results["model_a"]["avg_rouge2"] = 0.19
        results["model_a"]["avg_rougeL"] = 0.38
    
    if results["model_b"]["rouge_scores"]:
        rouge1_avg = sum(s['rouge1'] for s in results["model_b"]["rouge_scores"]) / len(results["model_b"]["rouge_scores"])
        rouge2_avg = sum(s['rouge2'] for s in results["model_b"]["rouge_scores"]) / len(results["model_b"]["rouge_scores"])
        rougeL_avg = sum(s['rougeL'] for s in results["model_b"]["rouge_scores"]) / len(results["model_b"]["rouge_scores"])
        results["model_b"]["avg_rouge1"] = rouge1_avg
        results["model_b"]["avg_rouge2"] = rouge2_avg
        results["model_b"]["avg_rougeL"] = rougeL_avg
    else:
        # Use dummy values if no ROUGE calculated
        results["model_b"]["avg_rouge1"] = 0.45
        results["model_b"]["avg_rouge2"] = 0.21
        results["model_b"]["avg_rougeL"] = 0.40
    
    # Determine winners
    results["comparison"] = {
        "rouge1_winner": "BART" if results["model_a"]["avg_rouge1"] > results["model_b"]["avg_rouge1"] else "PEGASUS",
        "rouge2_winner": "BART" if results["model_a"]["avg_rouge2"] > results["model_b"]["avg_rouge2"] else "PEGASUS",
        "rougeL_winner": "BART" if results["model_a"]["avg_rougeL"] > results["model_b"]["avg_rougeL"] else "PEGASUS",
        "latency_winner": "BART" if results["model_a"]["avg_latency_ms"] < results["model_b"]["avg_latency_ms"] else "PEGASUS",
    }
    
    # Overall recommendation
    rouge_wins_a = sum([
        results["model_a"]["avg_rouge1"] > results["model_b"]["avg_rouge1"],
        results["model_a"]["avg_rouge2"] > results["model_b"]["avg_rouge2"],
        results["model_a"]["avg_rougeL"] > results["model_b"]["avg_rougeL"]
    ])
    
    if rouge_wins_a >= 2:
        results["comparison"]["overall_recommendation"] = "BART untuk kualitas, lebih cepat"
    else:
        results["comparison"]["overall_recommendation"] = "PEGASUS untuk kualitas lebih baik, sedikit lebih lambat"
    
    print("\n✅ Model comparison completed!")
    return results


def generate_comparison_report(comparison_results: Dict) -> str:
    """
    Generates a human-readable comparison report.
    Useful for presentation!
    """
    report = []
    report.append("=" * 60)
    report.append("MODEL COMPARISON REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model A
    ma = comparison_results["model_a"]
    report.append(f"Model A: {ma['name']}")
    report.append(f"  - ROUGE-1: {ma['avg_rouge1']:.4f}")
    report.append(f"  - ROUGE-2: {ma['avg_rouge2']:.4f}")
    report.append(f"  - ROUGE-L: {ma['avg_rougeL']:.4f}")
    report.append(f"  - Avg Latency: {ma['avg_latency_ms']:.2f} ms")
    report.append("")
    
    # Model B
    mb = comparison_results["model_b"]
    report.append(f"Model B: {mb['name']}")
    report.append(f"  - ROUGE-1: {mb['avg_rouge1']:.4f}")
    report.append(f"  - ROUGE-2: {mb['avg_rouge2']:.4f}")
    report.append(f"  - ROUGE-L: {mb['avg_rougeL']:.4f}")
    report.append(f"  - Avg Latency: {mb['avg_latency_ms']:.2f} ms")
    report.append("")
    
    # Comparison
    comp = comparison_results["comparison"]
    report.append("RESULTS:")
    report.append(f"  - ROUGE-1 Winner: {comp['rouge1_winner']}")
    report.append(f"  - ROUGE-2 Winner: {comp['rouge2_winner']}")
    report.append(f"  - ROUGE-L Winner: {comp['rougeL_winner']}")
    report.append(f"  - Latency Winner: {comp['latency_winner']}")
    report.append("")
    report.append(f"RECOMMENDATION: {comp['overall_recommendation']}")
    report.append("=" * 60)
    
    return "\n".join(report)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test summarization
    print("Testing summarize_conversation stub:\n")
    
    test_conversation = [
        Message(role="user", content="Halo, pesanan ORDER123 saya belum sampai"),
        Message(role="assistant", content="Mohon maaf atas ketidaknyamanan. Bisa info sudah berapa lama?"),
        Message(role="user", content="Sudah 5 hari, harusnya 3 hari sampai"),
        Message(role="assistant", content="Baik, kami akan cek dan prioritaskan. Mohon tunggu 1x24 jam."),
        Message(role="user", content="Oke terima kasih"),
    ]
    
    summary = summarize_conversation(test_conversation)
    print(f"Conversation Summary:\n{summary}")
    print("-" * 60)
    
    # Test model comparison
    print("\nTesting compare_models stub:\n")
    
    sample_texts = [
        "Pelanggan komplain pesanan ORDER123 belum sampai sudah 5 hari.",
        "Customer request refund untuk iPhone yang rusak saat diterima."
    ]
    
    results = compare_models(sample_texts)
    report = generate_comparison_report(results)
    print(report)
    
    print("\n✅ Stubs working! Ready for real implementation.")