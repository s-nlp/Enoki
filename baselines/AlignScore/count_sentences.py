#!/usr/bin/env python3
"""
Script to count sentences with errors and total sentences in FactCheckBench data.

This script:
1. Loads FactCheckBench data (contexts and claims)
2. Extracts all sentences from the data
3. Counts total sentences and sentences with errors (NA labels)
4. Outputs the counts

Usage:
    python count_sentences.py [--data_path DATA_PATH]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

def norm_gold_label_factbench(x) -> Optional[bool]:
    """Normalize gold label from FactCheckBench data."""
    if x is True or x is False:
        return bool(x)
    if x is None:
        return None
    if isinstance(x, str) and x.strip().upper() == "NA":
        return None
    return None

def flatten_evidence(x: Any) -> List[str]:
    """Flatten evidence data into a list of strings."""
    out: List[str] = []

    def rec(v):
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, (list, tuple)):
            for t in v:
                rec(t)
        elif isinstance(v, dict):
            out.append(json.dumps(v, ensure_ascii=False))
        else:
            out.append(str(v))

    rec(x)
    return out

def count_sentences(data_path: str) -> Dict[str, Any]:
    """
    Count sentences with errors and total sentences from FactCheckBench data.
    
    Args:
        data_path: Path to FactCheckBench JSONL file
        
    Returns:
        Dictionary with counts and statistics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {data_path}")
    
    # Counters
    total_sentences = 0
    sentences_with_na_labels = 0
    sentences_without_evidence = 0
    valid_sentences = 0
    processed_sentences = []
    
    # Document-level counters
    total_docs = 0
    docs_without_errors = 0
    docs_with_na_labels = 0
    docs_without_evidence = 0
    docs_with_mixed_errors = 0
    
    # Load data
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Process each sample
    for sample_idx, sample in enumerate(samples):
        total_docs += 1
        
        # Track document-level error status
        doc_has_na_labels = False
        doc_has_no_evidence = False
        doc_has_valid_sentences = False
        
        # The 'sentences' dict contains the individual claims
        sentences_data = sample.get("sentences", {})
        
        for sent_key, sent_data in sentences_data.items():
            total_sentences += 1
            
            # Extract the claim (the sentence itself)
            sentence_text = sent_data.get("text", "").strip()
            
            # Check for NA labels
            gold_label = norm_gold_label_factbench(sent_data.get("sentence_factuality_label"))
            if gold_label is None:
                sentences_with_na_labels += 1
                doc_has_na_labels = True
                continue
            
            # Check for evidence
            human_ev = sent_data.get("human_evidence", [])
            auto_ev = sent_data.get("auto_evidence", [])
            
            ev_chunks_sentence: List[str] = []
            ev_chunks_sentence += flatten_evidence(human_ev)
            ev_chunks_sentence += flatten_evidence(auto_ev)
            
            if len(ev_chunks_sentence) == 0:
                sentences_without_evidence += 1
                doc_has_no_evidence = True
                continue
            
            # This is a valid sentence that would be processed
            valid_sentences += 1
            doc_has_valid_sentences = True
            processed_sentences.append({
                'sample_idx': sample_idx,
                'sentence_key': sent_key,
                'text': sentence_text,
                'label': gold_label,
                'evidence_count': len(ev_chunks_sentence)
            })
        
        # Determine document error status
        if not doc_has_na_labels and not doc_has_no_evidence and doc_has_valid_sentences:
            docs_without_errors += 1
        elif doc_has_na_labels and not doc_has_no_evidence:
            docs_with_na_labels += 1
        elif not doc_has_na_labels and doc_has_no_evidence:
            docs_without_evidence += 1
        else:
            docs_with_mixed_errors += 1
    
    # Calculate statistics
    results = {
        'total_sentences': total_sentences,
        'sentences_with_na_labels': sentences_with_na_labels,
        'sentences_without_evidence': sentences_without_evidence,
        'valid_sentences': valid_sentences,
        'processed_sentences': valid_sentences,  # Same as valid_sentences in this context
        'error_rate_na_labels': sentences_with_na_labels / total_sentences if total_sentences > 0 else 0,
        'error_rate_no_evidence': sentences_without_evidence / total_sentences if total_sentences > 0 else 0,
        'processing_rate': valid_sentences / total_sentences if total_sentences > 0 else 0,
        
        # Document-level statistics
        'total_docs': total_docs,
        'docs_without_errors': docs_without_errors,
        'docs_with_na_labels': docs_with_na_labels,
        'docs_without_evidence': docs_without_evidence,
        'docs_with_mixed_errors': docs_with_mixed_errors,
        'doc_error_rate_na_labels': docs_with_na_labels / total_docs if total_docs > 0 else 0,
        'doc_error_rate_no_evidence': docs_without_evidence / total_docs if total_docs > 0 else 0,
        'doc_error_rate_mixed': docs_with_mixed_errors / total_docs if total_docs > 0 else 0,
        'doc_processing_rate': docs_without_errors / total_docs if total_docs > 0 else 0,
        
        'processed_sentence_details': processed_sentences
    }
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Count sentences with errors and total sentences in FactCheckBench data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/factcheck-GPT-benchmark.jsonl",
        help="Path to FactCheckBench JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Count sentences
    logger.info("Starting sentence counting")
    results = count_sentences(args.data_path)
    
    # Print results
    print("\n" + "="*60)
    print("SENTENCE COUNTING RESULTS")
    print("="*60)
    print("Sentence-level Statistics:")
    print(f"  Total sentences: {results['total_sentences']:,}")
    print(f"  Sentences with NA labels: {results['sentences_with_na_labels']:,}")
    print(f"  Sentences without evidence: {results['sentences_without_evidence']:,}")
    print(f"  Valid sentences (would be processed): {results['valid_sentences']:,}")
    print(f"")
    print("Sentence Error Rates:")
    print(f"  NA labels rate: {results['error_rate_na_labels']:.2%}")
    print(f"  No evidence rate: {results['error_rate_no_evidence']:.2%}")
    print(f"  Processing rate: {results['processing_rate']:.2%}")
    print(f"")
    print("Document-level Statistics:")
    print(f"  Total documents: {results['total_docs']:,}")
    print(f"  Documents without errors: {results['docs_without_errors']:,}")
    print(f"  Documents with NA labels only: {results['docs_with_na_labels']:,}")
    print(f"  Documents without evidence only: {results['docs_without_evidence']:,}")
    print(f"  Documents with mixed errors: {results['docs_with_mixed_errors']:,}")
    print(f"")
    print("Document Error Rates:")
    print(f"  NA labels rate: {results['doc_error_rate_na_labels']:.2%}")
    print(f"  No evidence rate: {results['doc_error_rate_no_evidence']:.2%}")
    print(f"  Mixed errors rate: {results['doc_error_rate_mixed']:.2%}")
    print(f"  Processing rate: {results['doc_processing_rate']:.2%}")
    print("="*60)
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            # Remove detailed sentence info for file output to keep it manageable
            save_results = {k: v for k, v in results.items() if k != 'processed_sentence_details'}
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()