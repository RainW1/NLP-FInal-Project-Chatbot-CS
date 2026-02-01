"""
Model Evaluation & Comparison Module
====================================

Compares Pre-trained vs Fine-tuned models using proper NLP metrics:
- ROUGE (Rouge-1, Rouge-2, Rouge-L)
- BLEU Score
- Response Relevance
- Latency

SETUP:
pip install transformers torch rouge_score nltk
"""

import os
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import json

# ============================================================
# IMPORTS
# ============================================================

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ Please install: pip install rouge_score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ Please install: pip install nltk")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Please install: pip install transformers torch")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Please install: pip install groq")


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Models
PRETRAINED_MODEL = "llama-3.1-8b-instant"  # Via Groq
FINETUNED_MODEL = "google/flan-t5-base"    # Fine-tuned on instructions


# ============================================================
# TEST DATASET WITH REFERENCE ANSWERS
# ============================================================

EVALUATION_DATASET = [
    {
        "input": "My order hasn't arrived yet. It's been 5 days.",
        "reference": "I apologize for the delay with your order. Standard shipping typically takes 3-5 business days. Please provide your order number so I can track it and check the current status for you.",
        "category": "shipping"
    },
    {
        "input": "How do I get a refund?",
        "reference": "To request a refund, please go to My Orders section, select the item, and click Request Return. Items must be returned within 30 days in original condition. Refunds are processed within 5-7 business days after we receive the item.",
        "category": "refund"
    },
    {
        "input": "What payment methods do you accept?",
        "reference": "We accept Credit and Debit cards including Visa, MasterCard, and American Express. We also accept PayPal, Apple Pay, Google Pay, and bank transfers. Installment options are available through Klarna and Afterpay for orders over $100.",
        "category": "payment"
    },
    {
        "input": "I received a damaged product.",
        "reference": "I'm sorry to hear that your product arrived damaged. Please take photos of the damage and submit a complaint through our app within 48 hours. We will arrange a free replacement or full refund for you.",
        "category": "complaint"
    },
    {
        "input": "How long does shipping take?",
        "reference": "Standard shipping takes 3-5 business days for domestic orders and 7-14 business days for international orders. Express shipping with 1-2 day delivery is available for an additional $15.",
        "category": "shipping"
    },
    {
        "input": "Can I cancel my order?",
        "reference": "Yes, you can cancel your order if it hasn't been shipped yet. Please provide your order number and I'll cancel it immediately. If the order has already shipped, you can refuse delivery or return the item.",
        "category": "order"
    },
    {
        "input": "I want to return this item.",
        "reference": "Of course! You can return items within 30 days of delivery. Items must be unused and in original condition with tags attached. Go to My Orders, select the item, and click Request Return to get a return shipping label.",
        "category": "return"
    },
    {
        "input": "When will I receive my refund?",
        "reference": "Refunds are processed within 5-7 business days after we receive and verify your returned item. It may take an additional 3-5 days to appear in your account depending on your bank.",
        "category": "refund"
    },
    {
        "input": "Is my payment information secure?",
        "reference": "Absolutely! We use SSL encryption to protect all transactions. We are PCI-DSS compliant and never store your complete credit card information on our servers. Your payment data is completely secure.",
        "category": "payment"
    },
    {
        "input": "Hello, I need help.",
        "reference": "Hello! Welcome to our customer service. I'm here to help you with any questions about orders, shipping, returns, or products. How can I assist you today?",
        "category": "greeting"
    },
]


# ============================================================
# EVALUATION METRICS
# ============================================================

class EvaluationMetrics:
    """Calculate NLP evaluation metrics"""
    
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        self.smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE:
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 4),
            "rouge2": round(scores['rouge2'].fmeasure, 4),
            "rougeL": round(scores['rougeL'].fmeasure, 4)
        }
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score"""
        if not NLTK_AVAILABLE:
            return 0.0
        
        reference_tokens = [reference.lower().split()]
        prediction_tokens = prediction.lower().split()
        
        try:
            score = sentence_bleu(
                reference_tokens, 
                prediction_tokens,
                smoothing_function=self.smoothing
            )
            return round(score, 4)
        except:
            return 0.0
    
    def calculate_precision_recall_f1(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate token-level precision, recall, and F1
        Based on word overlap
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        # Common tokens
        common = pred_tokens.intersection(ref_tokens)
        
        # Precision: correct predictions / all predictions
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        
        # Recall: correct predictions / all reference tokens
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    
    def evaluate(self, prediction: str, reference: str) -> Dict:
        """Calculate all metrics"""
        rouge = self.calculate_rouge(prediction, reference)
        bleu = self.calculate_bleu(prediction, reference)
        prf = self.calculate_precision_recall_f1(prediction, reference)
        
        return {
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bleu": bleu,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"]
        }


# ============================================================
# MODEL LOADERS
# ============================================================

_finetuned_model = None
_finetuned_tokenizer = None

def load_finetuned_model():
    """Load fine-tuned model (singleton)"""
    global _finetuned_model, _finetuned_tokenizer
    
    if _finetuned_model is None:
        print(f"🔄 Loading fine-tuned model: {FINETUNED_MODEL}...")
        _finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
        _finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL)
        print("✅ Fine-tuned model loaded!")
    
    return _finetuned_model, _finetuned_tokenizer


# ============================================================
# MODEL RESPONSE GENERATORS
# ============================================================

def generate_pretrained_response(prompt: str) -> Tuple[str, float]:
    """Generate response using pre-trained Llama via Groq"""
    if not GROQ_AVAILABLE or GROQ_API_KEY == "YOUR_API_KEY_HERE":
        return "Groq API not configured", 0
    
    client = Groq(api_key=GROQ_API_KEY)
    
    system_prompt = """You are a helpful e-commerce customer service assistant.
Respond helpfully and concisely. Provide accurate information about orders, shipping, returns, and payments."""

    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=PRETRAINED_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        latency = (time.time() - start_time) * 1000
        return response.choices[0].message.content.strip(), latency
    
    except Exception as e:
        return f"Error: {e}", 0


def generate_finetuned_response(prompt: str) -> Tuple[str, float]:
    """Generate response using fine-tuned Flan-T5"""
    if not TRANSFORMERS_AVAILABLE:
        return "Transformers not available", 0
    
    try:
        model, tokenizer = load_finetuned_model()
        
        input_text = f"Answer as a helpful customer service agent: {prompt}"
        
        start_time = time.time()
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
        
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = (time.time() - start_time) * 1000
        
        return response, latency
    
    except Exception as e:
        return f"Error: {e}", 0


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def run_evaluation() -> Dict:
    """
    Run full evaluation comparing both models
    """
    metrics = EvaluationMetrics()
    
    pretrained_results = []
    finetuned_results = []
    
    print("=" * 70)
    print("MODEL EVALUATION: Pre-trained vs Fine-tuned")
    print("=" * 70)
    print(f"\n🅰️ Pre-trained: {PRETRAINED_MODEL}")
    print(f"🅱️ Fine-tuned: {FINETUNED_MODEL}")
    print(f"📊 Test samples: {len(EVALUATION_DATASET)}")
    print("=" * 70)
    
    for i, sample in enumerate(EVALUATION_DATASET, 1):
        print(f"\n📝 Test {i}/{len(EVALUATION_DATASET)}: {sample['category'].upper()}")
        print(f"Input: {sample['input'][:50]}...")
        
        # Generate responses
        pretrained_resp, pretrained_lat = generate_pretrained_response(sample['input'])
        finetuned_resp, finetuned_lat = generate_finetuned_response(sample['input'])
        
        # Calculate metrics
        pretrained_metrics = metrics.evaluate(pretrained_resp, sample['reference'])
        pretrained_metrics['latency_ms'] = pretrained_lat
        pretrained_metrics['response'] = pretrained_resp
        pretrained_results.append(pretrained_metrics)
        
        finetuned_metrics = metrics.evaluate(finetuned_resp, sample['reference'])
        finetuned_metrics['latency_ms'] = finetuned_lat
        finetuned_metrics['response'] = finetuned_resp
        finetuned_results.append(finetuned_metrics)
        
        print(f"  🅰️ ROUGE-L: {pretrained_metrics['rougeL']:.3f} | F1: {pretrained_metrics['f1']:.3f}")
        print(f"  🅱️ ROUGE-L: {finetuned_metrics['rougeL']:.3f} | F1: {finetuned_metrics['f1']:.3f}")
    
    # Calculate averages
    def calc_average(results: List[Dict], key: str) -> float:
        return round(sum(r[key] for r in results) / len(results), 4)
    
    summary = {
        "num_samples": len(EVALUATION_DATASET),
        "pretrained": {
            "model": PRETRAINED_MODEL,
            "avg_rouge1": calc_average(pretrained_results, "rouge1"),
            "avg_rouge2": calc_average(pretrained_results, "rouge2"),
            "avg_rougeL": calc_average(pretrained_results, "rougeL"),
            "avg_bleu": calc_average(pretrained_results, "bleu"),
            "avg_precision": calc_average(pretrained_results, "precision"),
            "avg_recall": calc_average(pretrained_results, "recall"),
            "avg_f1": calc_average(pretrained_results, "f1"),
            "avg_latency_ms": calc_average(pretrained_results, "latency_ms"),
        },
        "finetuned": {
            "model": FINETUNED_MODEL,
            "avg_rouge1": calc_average(finetuned_results, "rouge1"),
            "avg_rouge2": calc_average(finetuned_results, "rouge2"),
            "avg_rougeL": calc_average(finetuned_results, "rougeL"),
            "avg_bleu": calc_average(finetuned_results, "bleu"),
            "avg_precision": calc_average(finetuned_results, "precision"),
            "avg_recall": calc_average(finetuned_results, "recall"),
            "avg_f1": calc_average(finetuned_results, "f1"),
            "avg_latency_ms": calc_average(finetuned_results, "latency_ms"),
        }
    }
    
    # Determine winners
    summary["winners"] = {
        "rouge1": "Pre-trained" if summary["pretrained"]["avg_rouge1"] > summary["finetuned"]["avg_rouge1"] else "Fine-tuned",
        "rouge2": "Pre-trained" if summary["pretrained"]["avg_rouge2"] > summary["finetuned"]["avg_rouge2"] else "Fine-tuned",
        "rougeL": "Pre-trained" if summary["pretrained"]["avg_rougeL"] > summary["finetuned"]["avg_rougeL"] else "Fine-tuned",
        "bleu": "Pre-trained" if summary["pretrained"]["avg_bleu"] > summary["finetuned"]["avg_bleu"] else "Fine-tuned",
        "f1": "Pre-trained" if summary["pretrained"]["avg_f1"] > summary["finetuned"]["avg_f1"] else "Fine-tuned",
        "latency": "Pre-trained" if summary["pretrained"]["avg_latency_ms"] < summary["finetuned"]["avg_latency_ms"] else "Fine-tuned",
    }
    
    # Print summary
    print_evaluation_summary(summary)
    
    return summary


def print_evaluation_summary(summary: Dict):
    """Print formatted evaluation summary"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "EVALUATION RESULTS" + " " * 30 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + " " * 68 + "║")
    
    # Table header
    print("║  {:<20} {:>15} {:>15} {:>12}  ║".format(
        "Metric", "Pre-trained", "Fine-tuned", "Winner"))
    print("║  " + "-" * 64 + "  ║")
    
    # Metrics rows
    metrics = [
        ("ROUGE-1", "avg_rouge1"),
        ("ROUGE-2", "avg_rouge2"),
        ("ROUGE-L", "avg_rougeL"),
        ("BLEU", "avg_bleu"),
        ("Precision", "avg_precision"),
        ("Recall", "avg_recall"),
        ("F1 Score", "avg_f1"),
    ]
    
    for label, key in metrics:
        pre_val = summary["pretrained"][key]
        fine_val = summary["finetuned"][key]
        winner_key = key.replace("avg_", "")
        winner = summary["winners"].get(winner_key, "")
        
        # Highlight winner
        winner_symbol = "🏆" if winner else ""
        
        print("║  {:<20} {:>15.4f} {:>15.4f} {:>12}  ║".format(
            label, pre_val, fine_val, winner))
    
    print("║  " + "-" * 64 + "  ║")
    
    # Latency
    print("║  {:<20} {:>12.2f} ms {:>12.2f} ms {:>12}  ║".format(
        "Avg Latency",
        summary["pretrained"]["avg_latency_ms"],
        summary["finetuned"]["avg_latency_ms"],
        summary["winners"]["latency"]))
    
    print("║" + " " * 68 + "║")
    print("╠" + "═" * 68 + "╣")
    
    # Conclusion
    pre_wins = sum(1 for v in summary["winners"].values() if v == "Pre-trained")
    fine_wins = sum(1 for v in summary["winners"].values() if v == "Fine-tuned")
    
    overall_winner = "Pre-trained" if pre_wins > fine_wins else "Fine-tuned"
    
    print("║" + " " * 68 + "║")
    print("║  CONCLUSION:" + " " * 55 + "║")
    print("║  • Pre-trained wins: {} metrics".format(pre_wins) + " " * 45 + "║")
    print("║  • Fine-tuned wins: {} metrics".format(fine_wins) + " " * 46 + "║")
    print("║  • Overall Winner: {} 🏆".format(overall_winner) + " " * (47 - len(overall_winner)) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")


def save_results_to_json(summary: Dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to {filename}")


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Starting Model Evaluation...\n")
    
    # Run evaluation
    results = run_evaluation()
    
    # Save results
    save_results_to_json(results)
    
    print("\n✅ Evaluation complete!")