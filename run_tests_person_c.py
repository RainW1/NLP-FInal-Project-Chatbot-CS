"""
Test Runner for Person C - NLP Module
Menggunakan data test dari file JSON yang terpisah
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interfaces import Message
from src.nlp.ner import extract_entities
from src.nlp.summarizer import summarize_conversation, compare_models, generate_comparison_report

def load_test_data():
    """Load test data dari file JSON"""
    with open('data/test_data_person_c.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def test_ner(test_data):
    """Test NER dengan data dari file"""
    print("="*70)
    print("TEST 1: NAMED ENTITY RECOGNITION (NER)")
    print("="*70)
    
    ner_cases = test_data['ner_test_cases']
    passed = 0
    failed = 0
    
    for case in ner_cases:
        print(f"\n[{case['id']}] {case['text']}")
        result = extract_entities(case['text'])
        
        expected = case['expected']
        all_match = True
        
        # Check each expected entity
        if 'order_id' in expected:
            if result.order_id == expected['order_id']:
                print(f"  ✅ Order ID: {result.order_id}")
            else:
                print(f"  ❌ Order ID: Got '{result.order_id}', Expected '{expected['order_id']}'")
                all_match = False
        
        if 'product_name' in expected:
            if result.product_name and expected['product_name'].lower() in result.product_name.lower():
                print(f"  ✅ Product: {result.product_name}")
            else:
                print(f"  ❌ Product: Got '{result.product_name}', Expected '{expected['product_name']}'")
                all_match = False
        
        if 'date' in expected:
            if result.date and expected['date'].lower() in result.date.lower():
                print(f"  ✅ Date: {result.date}")
            else:
                print(f"  ❌ Date: Got '{result.date}', Expected '{expected['date']}'")
                all_match = False
        
        if 'amount' in expected:
            if result.amount == expected['amount']:
                print(f"  ✅ Amount: {result.amount}")
            else:
                print(f"  ❌ Amount: Got '{result.amount}', Expected '{expected['amount']}'")
                all_match = False
        
        if 'person_name' in expected:
            if result.person_name and expected['person_name'] in result.person_name:
                print(f"  ✅ Person: {result.person_name}")
            else:
                print(f"  ❌ Person: Got '{result.person_name}', Expected '{expected['person_name']}'")
                all_match = False
        
        if all_match:
            passed += 1
            print(f"  ✅ PASS")
        else:
            failed += 1
            print(f"  ❌ FAIL")
    
    print(f"\n{'='*70}")
    print(f"NER Test Results: {passed} passed, {failed} failed out of {len(ner_cases)} tests")
    print(f"{'='*70}")
    return passed, failed

def test_summarization(test_data):
    """Test Summarization dengan data dari file"""
    print("\n" + "="*70)
    print("TEST 2: CONVERSATION SUMMARIZATION")
    print("="*70)
    
    conv_cases = test_data['conversation_test_cases']
    passed = 0
    failed = 0
    
    for case in conv_cases:
        print(f"\n[{case['id']}] {case['title']}")
        print(f"Messages: {len(case['messages'])}")
        
        # Convert to Message objects
        messages = [Message(role=m['role'], content=m['content']) for m in case['messages']]
        
        # Generate summary
        summary = summarize_conversation(messages)
        print(f"\nGenerated Summary:\n{summary}\n")
        
        # Check if expected keywords are in summary
        expected_keywords = case['expected_summary_keywords']
        found_keywords = [kw for kw in expected_keywords if kw.lower() in summary.lower()]
        
        print(f"Expected keywords: {expected_keywords}")
        print(f"Found keywords: {found_keywords}")
        
        if len(found_keywords) >= len(expected_keywords) / 2:  # At least 50% keywords found
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL - Not enough keywords found")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Summarization Test Results: {passed} passed, {failed} failed out of {len(conv_cases)} tests")
    print(f"{'='*70}")
    return passed, failed

def test_model_comparison(test_data):
    """Test Model Comparison dengan data dari file"""
    print("\n" + "="*70)
    print("TEST 3: MODEL COMPARISON (BART vs PEGASUS)")
    print("="*70)
    
    comp_samples = test_data['model_comparison_samples']
    
    # Extract texts and references
    texts = [sample['text'] for sample in comp_samples]
    references = [sample['reference_summary'] for sample in comp_samples]
    
    print(f"\nRunning comparison on {len(texts)} samples...")
    print("⚠️  This may take a while (downloading/loading models)...\n")
    
    # Run comparison
    results = compare_models(texts, references)
    
    # Generate report
    report = generate_comparison_report(results)
    print(report)
    
    # Show sample summaries
    print("\n" + "="*70)
    print("SAMPLE SUMMARIES")
    print("="*70)
    
    for i, sample in enumerate(comp_samples[:3]):  # Show first 3
        print(f"\n[{sample['id']}]")
        print(f"Reference: {sample['reference_summary']}")
        if i < len(results['model_a']['summaries']):
            print(f"BART: {results['model_a']['summaries'][i]}")
        if i < len(results['model_b']['summaries']):
            print(f"PEGASUS: {results['model_b']['summaries'][i]}")
    
    return True

if __name__ == "__main__":
    print("="*70)
    print("PERSON C - NLP MODULE TEST SUITE")
    print("Using data from: data/test_data_person_c.json")
    print("="*70)
    
    # Load test data
    test_data = load_test_data()
    print(f"\nLoaded test data:")
    print(f"  - {len(test_data['ner_test_cases'])} NER test cases")
    print(f"  - {len(test_data['conversation_test_cases'])} conversation test cases")
    print(f"  - {len(test_data['model_comparison_samples'])} comparison samples")
    
    # Run tests
    print("\n" + "="*70)
    choice = input("\nPilih test yang mau dijalankan:\n1. NER only\n2. Summarization only\n3. Model Comparison only\n4. All tests\n\nPilihan (1-4): ")
    
    total_passed = 0
    total_failed = 0
    
    if choice in ['1', '4']:
        passed, failed = test_ner(test_data)
        total_passed += passed
        total_failed += failed
    
    if choice in ['2', '4']:
        passed, failed = test_summarization(test_data)
        total_passed += passed
        total_failed += failed
    
    if choice in ['3', '4']:
        test_model_comparison(test_data)
    
    # Final summary
    if choice in ['1', '2', '4']:
        print("\n" + "="*70)
        print("FINAL TEST SUMMARY")
        print("="*70)
        print(f"Total Passed: {total_passed}")
        print(f"Total Failed: {total_failed}")
        success_rate = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        print("="*70)
