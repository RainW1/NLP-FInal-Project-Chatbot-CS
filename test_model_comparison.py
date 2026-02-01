"""
Test script for model comparison module.
Run this to compare BART vs PEGASUS summarization models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp.summarizer import compare_models, generate_comparison_report

# Sample conversation texts for testing
sample_conversations = [
    """Pelanggan: Halo, pesanan saya ORDER123 belum sampai sudah 5 hari.
Customer Service: Mohon maaf atas ketidaknyamanan ini. Bisa tolong info kapan order dibuat?
Pelanggan: Tanggal 15 Januari kemarin. Harusnya 3 hari sampai.
Customer Service: Baik, saya akan cek status pengiriman Anda dan prioritaskan. Mohon tunggu 1x24 jam ya.
Pelanggan: Ok terima kasih.""",
    
    """Pelanggan: Saya mau komplain, iPhone 15 Pro Max yang saya terima rusak layarnya.
Customer Service: Mohon maaf mendengar hal ini. Apakah bisa kirim foto produk yang rusak?
Pelanggan: Sudah saya kirim via email. Nomor order INV-20240120001.
Customer Service: Terima kasih. Kami akan proses refund atau replacement. Pilih yang mana?
Pelanggan: Mau replacement saja.
Customer Service: Baik, kami akan kirim unit baru. Estimasi 2-3 hari kerja.""",

    """Pelanggan: Gimana cara tracking pesanan?
Customer Service: Bisa cek di halaman 'My Orders' dengan login ke akun Anda.
Pelanggan: Sudah cek tapi statusnya stuck di 'Processing'.
Customer Service: Boleh info nomor ordernya?
Pelanggan: ORDER789456
Customer Service: Baik saya cek dulu ya. Pesanan sedang dikemas oleh seller, nanti ada update resi dalam 6 jam."""
]

# Optional: Reference summaries for ROUGE calculation
reference_summaries = [
    "Pelanggan komplain pesanan ORDER123 terlambat 5 hari. CS akan prioritaskan pengecekan dalam 24 jam.",
    "Pelanggan terima iPhone rusak untuk order INV-20240120001. CS akan kirim replacement dalam 2-3 hari.",
    "Pelanggan tanya cara tracking ORDER789456 yang stuck. CS jelaskan pesanan sedang dikemas, resi akan tersedia dalam 6 jam."
]

if __name__ == "__main__":
    print("="*70)
    print("MODEL COMPARISON TEST: BART vs PEGASUS")
    print("="*70)
    print()
    
    print("📝 Test Samples:")
    for i, sample in enumerate(sample_conversations, 1):
        print(f"\n{i}. {sample[:100]}...")
    
    print("\n" + "="*70)
    print("Starting comparison...")
    print("="*70)
    
    # Run comparison
    results = compare_models(
        text_samples=sample_conversations,
        reference_summaries=reference_summaries
    )
    
    # Generate and print report
    print("\n" + "="*70)
    report = generate_comparison_report(results)
    print(report)
    
    print("\n📊 Sample Summaries:")
    print("\nBart Summaries:")
    for i, summary in enumerate(results["model_a"]["summaries"][:2], 1):
        print(f"  {i}. {summary}")
    
    print("\nPEGASUS Summaries:")
    for i, summary in enumerate(results["model_b"]["summaries"][:2], 1):
        print(f"  {i}. {summary}")
    
    print("\n✅ Model comparison test completed!")
