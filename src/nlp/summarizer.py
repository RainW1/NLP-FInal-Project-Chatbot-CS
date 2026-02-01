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
# STUB IMPLEMENTATION - Replace with real code!
# ============================================================

def summarize_conversation(messages: List[Message]) -> str:
    """
    STUB: Summarizes conversation history.
    
    TODO (Person C):
    1. Load summarization model (BART or PEGASUS)
    2. Format conversation into summarizable text
    3. Generate summary
    4. Post-process (clean, truncate if needed)
    
    Real implementation example:
    ```python
    from transformers import pipeline
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def summarize_conversation(messages: List[Message]) -> str:
        # Format conversation
        conversation_text = ""
        for msg in messages:
            role = "Pelanggan" if msg.role == "user" else "CS"
            conversation_text += f"{role}: {msg.content}\\n"
        
        # Generate summary
        if len(conversation_text) < 50:  # Too short to summarize
            return "Percakapan terlalu singkat untuk dirangkum."
        
        result = summarizer(
            conversation_text,
            max_length=100,
            min_length=30,
            do_sample=False
        )
        
        return result[0]["summary_text"]
    ```
    """
    if not messages:
        return "Tidak ada percakapan untuk dirangkum."
    
    if len(messages) < 2:
        return "Percakapan terlalu singkat untuk dirangkum."
    
    # STUB - Simple extractive summary based on keywords
    # Replace with real model!
    
    # Analyze conversation
    user_messages = [m.content for m in messages if m.role == "user"]
    bot_messages = [m.content for m in messages if m.role == "assistant"]
    
    # Extract key information
    topics = []
    if any("belum sampai" in m.lower() or "terlambat" in m.lower() for m in user_messages):
        topics.append("keterlambatan pengiriman")
    if any("rusak" in m.lower() or "cacat" in m.lower() for m in user_messages):
        topics.append("produk rusak/cacat")
    if any("refund" in m.lower() or "return" in m.lower() for m in user_messages):
        topics.append("permintaan refund/return")
    if any("status" in m.lower() or "tracking" in m.lower() for m in user_messages):
        topics.append("pengecekan status pesanan")
    
    # Extract order ID if mentioned
    import re
    order_ids = []
    for m in user_messages:
        matches = re.findall(r'ORDER[-]?\d+|ORD[-]?\d+|#\d{6,}', m, re.IGNORECASE)
        order_ids.extend(matches)
    
    # Build summary
    summary_parts = []
    
    # Opening
    summary_parts.append(f"Percakapan dengan {len(messages)} pesan.")
    
    # Main topic
    if topics:
        summary_parts.append(f"Pelanggan menghubungi terkait {', '.join(topics)}.")
    else:
        summary_parts.append("Pelanggan menghubungi untuk bertanya.")
    
    # Order ID
    if order_ids:
        summary_parts.append(f"Nomor pesanan yang disebutkan: {', '.join(set(order_ids))}.")
    
    # Resolution status (simple heuristic)
    if any("terima kasih" in m.lower() or "thanks" in m.lower() for m in user_messages):
        summary_parts.append("Pelanggan mengucapkan terima kasih (kemungkinan terselesaikan).")
    
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

def compare_models(text_samples: List[str], reference_summaries: List[str] = None) -> Dict:
    """
    STUB: Compares two summarization models.
    
    TODO (Person C):
    1. Load both models (BART and PEGASUS)
    2. Run summarization on all samples
    3. Calculate ROUGE scores if references available
    4. Measure latency
    5. Return comparison results
    
    Real implementation example:
    ```python
    from transformers import pipeline
    from rouge_score import rouge_scorer
    
    # Load models
    bart = pipeline("summarization", model="facebook/bart-large-cnn")
    pegasus = pipeline("summarization", model="google/pegasus-cnn_dailymail")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def compare_models(text_samples, reference_summaries=None):
        results = {
            "model_a": {"name": "BART", "summaries": [], "rouge_scores": [], "latencies": []},
            "model_b": {"name": "PEGASUS", "summaries": [], "rouge_scores": [], "latencies": []}
        }
        
        for i, text in enumerate(text_samples):
            # BART
            start = time.time()
            bart_summary = bart(text, max_length=100)[0]["summary_text"]
            results["model_a"]["latencies"].append((time.time() - start) * 1000)
            results["model_a"]["summaries"].append(bart_summary)
            
            # PEGASUS
            start = time.time()
            pegasus_summary = pegasus(text, max_length=100)[0]["summary_text"]
            results["model_b"]["latencies"].append((time.time() - start) * 1000)
            results["model_b"]["summaries"].append(pegasus_summary)
            
            # Calculate ROUGE if references available
            if reference_summaries:
                bart_scores = scorer.score(reference_summaries[i], bart_summary)
                pegasus_scores = scorer.score(reference_summaries[i], pegasus_summary)
                results["model_a"]["rouge_scores"].append(bart_scores)
                results["model_b"]["rouge_scores"].append(pegasus_scores)
        
        # Calculate averages
        results["model_a"]["avg_latency"] = sum(results["model_a"]["latencies"]) / len(text_samples)
        results["model_b"]["avg_latency"] = sum(results["model_b"]["latencies"]) / len(text_samples)
        
        return results
    ```
    """
    # STUB - Return dummy comparison results
    results = {
        "model_a": {
            "name": "BART (facebook/bart-large-cnn)",
            "avg_rouge1": 0.42,
            "avg_rouge2": 0.19,
            "avg_rougeL": 0.38,
            "avg_latency_ms": 245.5,
            "sample_summaries": [
                "Pelanggan komplain pengiriman terlambat.",
                "Request refund untuk produk rusak."
            ]
        },
        "model_b": {
            "name": "PEGASUS (google/pegasus-cnn_dailymail)",
            "avg_rouge1": 0.45,
            "avg_rouge2": 0.21,
            "avg_rougeL": 0.40,
            "avg_latency_ms": 312.3,
            "sample_summaries": [
                "Customer reported delayed delivery issue.",
                "Refund requested for damaged product."
            ]
        },
        "comparison": {
            "rouge1_winner": "PEGASUS",
            "rouge2_winner": "PEGASUS", 
            "rougeL_winner": "PEGASUS",
            "latency_winner": "BART",
            "overall_recommendation": "PEGASUS untuk kualitas, BART untuk kecepatan"
        },
        "num_samples": len(text_samples)
    }
    
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
