#!/usr/bin/env python3
"""
AlignScore evaluation script for FactCheckBench dataset.

This script:
1. Loads FactCheckBench data (contexts and claims)
2. Runs AlignScore on all context/claim pairs
3. Tests different hyperparameters (evaluation modes)
4. Computes metrics including AUROC
5. Outputs comprehensive results

Usage:
    python run_alignscore_factcheckbench.py [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

# Import AlignScore
try:
    from AlignScore.src.alignscore import AlignScore
except ImportError:
    raise ImportError(
        "AlignScore not found. Please ensure the AlignScore package is properly installed "
        "or the script is run from the correct directory."
    )


def norm_gold_label_factbench(x) -> Optional[bool]:
    if x is True or x is False:
        return bool(x)
    if x is None:
        return None
    if isinstance(x, str) and x.strip().upper() == "NA":
        return None
    return None

def flatten_evidence(x: Any) -> List[str]:
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

def safe_div(n, d):
    return n / d if d else 0.0

class FactCheckBenchAlignScoreEvaluator:
    """Evaluator for AlignScore on FactCheckBench dataset."""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        model_name: str = "roberta-large",
        batch_size: int = 32,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            data_path: Path to FactCheckBench JSONL file
            output_dir: Directory to save results
            model_name: AlignScore model name
            batch_size: Batch size for inference
            device: Device to run inference on
            checkpoint_path: Path to model checkpoint
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AlignScore configuration
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Initialize AlignScore
        self.scorer = AlignScore(
            model=model_name,
            batch_size=batch_size,
            device=device,
            ckpt_path=checkpoint_path,
        )
        
        # Evaluation modes to test
        self.evaluation_modes = [
            "nli_sp",      
            "nli",
            "bin_sp",
            "bin"
        ]
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.ground_truth = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'alignscore_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load FactCheckBench data from JSONL file.
        
        Returns:
            List of samples with contexts and claims
        """
        self.logger.info(f"Loading data from {self.data_path}")
        
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(samples)} samples")
        return samples

    def extract_contexts_and_claims(self, samples: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[bool]]:
        """
        Extract contexts, claims, and ground truth labels from samples.
        Prioritizes human_evidence, falls back to auto_evidence.
        """
        contexts = []
        claims = []
        labels = []
        
        for sample in samples:
            # The 'sentences' dict contains the individual claims
            sentences_data = sample.get("sentences", {})
            
            for sent_key, sent_data in sentences_data.items():
                # 1. Extract the claim (the sentence itself)
                sentence_text = sent_data.get("text", "").strip()
                if not sentence_text:
                    continue
                
                # 2. Extract Context (Human Evidence > Auto Evidence)
                # We join list items into a single string if they are provided as lists

                human_ev = sent_data.get("human_evidence", [])
                auto_ev = sent_data.get("auto_evidence", [])

                ev_chunks_sentence: List[str] = []
                ev_chunks_sentence += flatten_evidence(human_ev)
                ev_chunks_sentence += flatten_evidence(auto_ev)
                context = "\n\n".join(ev_chunks_sentence)


                gold_label = norm_gold_label_factbench(sent_data.get("sentence_factuality_label"))
                if gold_label is None:
                    self.logger.warning(f"NA")
                    continue

                if len(ev_chunks_sentence) == 0:
                    # Skip if there is no evidence to check against
                    self.logger.warning(f"No evidence found for sentence: {sent_key}")
                    continue

                # 3. Get ground truth label

                
                # Convert label to boolean
                if isinstance(gold_label, bool):
                    label = gold_label
                elif isinstance(gold_label, str):
                    label = gold_label.lower() == "true"
                else:
                    continue
                
                contexts.append(context)
                claims.append(sentence_text)
                labels.append(label)
        
        self.logger.info(f"Extracted {len(contexts)} context-claim pairs using evidence-based context.")
        return contexts, claims, labels

    def run_alignscore_evaluation(
        self, 
        contexts: List[str], 
        claims: List[str], 
        labels: List[bool],
        evaluation_mode: str
    ) -> Dict[str, Any]:
        """
        Run AlignScore evaluation for a specific evaluation mode.
        
        Args:
            contexts: List of context strings
            claims: List of claim strings
            labels: Ground truth labels
            evaluation_mode: AlignScore evaluation mode
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Running AlignScore evaluation with mode: {evaluation_mode}")
        
        # Set evaluation mode
        self.scorer = AlignScore(model=self.model_name, batch_size=self.batch_size, device=self.device, ckpt_path=self.checkpoint_path, evaluation_mode=evaluation_mode)
        
        # Run inference
        start_time = time.time()
        scores = self.scorer.score(contexts=contexts, claims=claims)
        inference_time = time.time() - start_time
        
        self.logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        # Convert scores to predictions
        # AlignScore returns scores where higher values indicate better alignment
        # We need to threshold these scores to get binary predictions
        predictions = self._scores_to_predictions(scores)
        
        # Calculate metrics
        metrics = self._calculate_metrics(labels, predictions, scores)
        
        # Store results
        result = {
            "evaluation_mode": evaluation_mode,
            "inference_time": inference_time,
            "scores": scores,
            "predictions": predictions,
            "ground_truth": labels,
            "metrics": metrics,
        }
        
        return result
    
    def _scores_to_predictions(self, scores: List[float], threshold: float = 0.5) -> List[bool]:
        """
        Convert AlignScore scores to binary predictions.
        
        Args:
            scores: List of AlignScore scores
            threshold: Threshold for binary classification
            
        Returns:
            List of boolean predictions
        """
        # Note: The exact interpretation of AlignScore scores depends on the evaluation mode
        # For NLI-based modes, higher scores typically indicate better alignment/support
        return [score >= threshold for score in scores]
    
    def run_comprehensive_evaluation(self, contexts: List[str], claims: List[str], labels: List[bool]):
        """
        Run comprehensive evaluation across all evaluation modes.
        
        Args:
            contexts: List of context strings
            claims: List of claim strings
            labels: Ground truth labels
        """
        self.logger.info("Starting comprehensive AlignScore evaluation")
        
        for mode in self.evaluation_modes:
            try:
                result = self.run_alignscore_evaluation(contexts, claims, labels, mode)
                self.results[mode] = result
                
                # Log results
                metrics = result["metrics"]
                self.logger.info(f"Results for {mode}:")
                self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                self.logger.info(f"  Precision: {metrics['precision']:.4f}")
                self.logger.info(f"  Recall: {metrics['recall']:.4f}")
                self.logger.info(f"  F1: {metrics['f1']:.4f}")
                self.logger.info(f"  AUROC: {metrics['auroc']:.4f}" if metrics['auroc'] else "  AUROC: N/A")
                
            except Exception as e:
                self.logger.error(f"Error evaluating mode {mode}: {e}")
                continue
    

    def save_results(self):
        """Save all results to files."""
        self.logger.info(f"Saving results to {self.output_dir}")
        
        # Save detailed results (JSON)
        results_file = self.output_dir / "alignscore_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            serializable_results = {}
            for mode, result in self.results.items():
                serializable_results[mode] = {
                    "evaluation_mode": result["evaluation_mode"],
                    "metrics": result["metrics"],
                    "inference_time": result["inference_time"]
                }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = self._create_summary()
        summary_file = self.output_dir / "alignscore_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary with new metric columns
        self._save_csv_summary(summary)
        
        self.logger.info("Results saved successfully")

    def _save_csv_summary(self, summary: Dict[str, Any]):
        """Save a CSV summary focused on FactCheckBench Macro metrics."""
        import csv
        
        csv_file = self.output_dir / "alignscore_summary.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Updated Header for Macro F1 and class-specific metrics
            writer.writerow([
                "Mode", 
                "F1 Macro", 
                "F1 (Not Supported)", 
                "Rec (Not Supported)", 
                "F1 (Supported)", 
                "AUROC", 
                "Accuracy",
                "TP", "FP", "FN", "TN"
            ])
            
            # Data rows
            for mode, m in summary["evaluation_modes"].items():
                cm = m["confusion_matrix"]
                writer.writerow([
                    mode,
                    f"{m['f1_macro']:.4f}",
                    f"{m['f1_not_supported']:.4f}",
                    f"{m['recall_not_supported']:.4f}",
                    f"{m['f1_supported']:.4f}",
                    f"{m['auroc']:.4f}" if m['auroc'] else "N/A",
                    f"{m['accuracy']:.4f}",
                    cm["TP"], cm["FP"], cm["FN"], cm["TN"]
                ])
            
            # Best results by Macro F1
            if summary["best_results"]:
                best = summary["best_results"]
                writer.writerow([])
                writer.writerow([f"Best Mode (by Macro F1): {best['mode']}"])

    def _create_summary(self) -> Dict[str, Any]:
        """Create a summary using the updated metric keys."""
        summary = {
            "model_config": {
                "model_name": self.model_name,
                "device": self.device,
            },
            "dataset": {
                "total_pairs": len(self.results[next(iter(self.results.keys()))]["ground_truth"]) if self.results else 0,
            },
            "evaluation_modes": {},
            "best_results": {},
        }
        
        best_f1_macro = -1.0
        best_mode = None
        
        for mode, result in self.results.items():
            metrics = result["metrics"]
            summary["evaluation_modes"][mode] = metrics
            
            if metrics["f1_macro"] > best_f1_macro:
                best_f1_macro = metrics["f1_macro"]
                best_mode = mode
        
        if best_mode:
            summary["best_results"] = {
                "mode": best_mode,
                "f1_macro": best_f1_macro,
                "metrics": summary["evaluation_modes"][best_mode],
            }
        
        return summary

    def _calculate_metrics(self, y_true: List[bool], y_pred: List[bool], y_scores: List[float]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics where 'Not Supported' is the positive class.
        y_true/y_pred: True means 'Supported', False means 'Not Supported'
        """
        cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        
        for gold_supported, pred_supported in zip(y_true, y_pred):
            # Logic: gold_pos = not gold_supported (Not Supported is Positive)
            gold_pos = not gold_supported
            pred_pos = not pred_supported
            
            if gold_pos and pred_pos:
                cm["TP"] += 1
            elif (not gold_pos) and pred_pos:
                cm["FP"] += 1
            elif gold_pos and (not pred_pos):
                cm["FN"] += 1
            else:
                cm["TN"] += 1

        # Calculate Precision, Recall, F1 for both classes
        p_ns, r_ns, f1_ns = self._prf1(cm["TP"], cm["FP"], cm["FN"])
        p_s, r_s, f1_s = self._prf1(cm["TN"], cm["FN"], cm["FP"])
        
        # Calculate AUROC (standard direction)
        try:
            # Note: roc_auc_score usually expects the score for the positive class.
            # If 'Not Supported' is positive, we use (1 - score)
            y_true_not_supported = [not label for label in y_true]
            inv_scores = [1.0 - s for s in y_scores]
            auroc = roc_auc_score(y_true_not_supported, inv_scores)
        except:
            auroc = 0.0

        return {
            "accuracy": (cm["TP"] + cm["TN"]) / sum(cm.values()) if sum(cm.values()) else 0,
            "precision_not_supported": p_ns,
            "recall_not_supported": r_ns,
            "f1_not_supported": f1_ns,
            "precision_supported": p_s,
            "recall_supported": r_s,
            "f1_supported": f1_s,
            "f1_macro": (f1_ns + f1_s) / 2.0,
            "auroc": float(auroc),
            "confusion_matrix": cm,
        }

    def _prf1(self, tp, fp, fn):
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
        return p, r, f1
    
    def run(self):
        """Main execution method."""
        self.logger.info("Starting AlignScore evaluation for FactCheckBench")
        
        # Load data
        samples = self.load_data()
        
        # Extract contexts and claims
        contexts, claims, labels = self.extract_contexts_and_claims(samples)
        
        # Store ground truth for summary
        self.ground_truth["all"] = labels
        
        # Run comprehensive evaluation
        self.run_comprehensive_evaluation(contexts, claims, labels)
        
        # Save results
        self.save_results()
        
        self.logger.info("AlignScore evaluation completed successfully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run AlignScore evaluation on FactCheckBench dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/factcheck-GPT-benchmark.jsonl",
        help="Path to FactCheckBench JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/alignscore_factcheckbench",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large",
        help="AlignScore model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="AlignScore/checkpoints/AlignScore-large.ckpt",
        help="Path to AlignScore checkpoint"
    )
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = FactCheckBenchAlignScoreEvaluator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()

# python run_alignscore_factcheckbench.py \
# --data_path "../factcheck-GPT-benchmark.jsonl" \
# --output_dir "results" \
# --model_name "roberta-large" \
# --batch_size 32 \
# --device "cuda" \
# --checkpoint_path "checkpoints/AlignScore-large.ckpt"


# python run_alignscore_factcheckbench.py \
# --data_path "../felm_with_ref_text/science.jsonl" \
# --output_dir "results/felm/science" \
# --model_name "roberta-large" \
# --batch_size 32 \
# --device "cuda" \
# --checkpoint_path "checkpoints/AlignScore-large.ckpt"

# python run_alignscore_factcheckbench.py \
# --data_path "../felm_with_ref_text/wk.jsonl" \
# --output_dir "results/felm/wk" \
# --model_name "roberta-large" \
# --batch_size 32 \
# --device "cuda" \
# --checkpoint_path "checkpoints/AlignScore-large.ckpt"

# python run_alignscore_factcheckbench.py \
# --data_path "../felm_with_ref_text/writing_rec.jsonl" \
# --output_dir "results/felm/writing_rec" \
# --model_name "roberta-large" \
# --batch_size 32 \
# --device "cuda" \
# --checkpoint_path "checkpoints/AlignScore-large.ckpt"
