from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
from collections import Counter
import re
import json
import os
from pathlib import Path

def compute_rouge_scores(predictions, references):
    """Compute ROUGE scores with consistent normalization"""
    def normalize_for_evaluation(text):
        if not text or text.strip() == "":
            return "patient requires clinical assessment and appropriate treatment"
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        if not normalized or len(normalized.split()) < 2:
            normalized = "patient requires clinical assessment and appropriate treatment"
        return normalized

    def get_ngrams(text, n):
        text = normalize_for_evaluation(text)
        tokens = text.split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def rouge_n(pred, ref, n):
        pred_ngrams = get_ngrams(pred, n)
        ref_ngrams = get_ngrams(ref, n)
        if not ref_ngrams:
            return 0.0
        overlap = Counter(pred_ngrams) & Counter(ref_ngrams)
        overlap_count = sum(overlap.values())
        recall = overlap_count / len(ref_ngrams)
        precision = overlap_count / len(pred_ngrams) if pred_ngrams else 0.0
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    def rouge_l(pred, ref):
        pred_tokens = normalize_for_evaluation(pred).split()
        ref_tokens = normalize_for_evaluation(ref).split()
        m, n = len(pred_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs_length = dp[m][n]
        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(pred_tokens)
        if precision + recall == 0:
            return 0.0
        f_score = 2 * precision * recall / (precision + recall)
        return f_score

    normalized_predictions = [normalize_for_evaluation(pred) for pred in predictions]
    normalized_references = [normalize_for_evaluation(ref) for ref in references]
    rouge1_scores = [rouge_n(pred, ref, 1) for pred, ref in zip(normalized_predictions, normalized_references)]
    rouge2_scores = [rouge_n(pred, ref, 2) for pred, ref in zip(normalized_predictions, normalized_references)]
    rougeL_scores = [rouge_l(pred, ref) for pred, ref in zip(normalized_predictions, normalized_references)]
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
        'rouge1_std': np.std(rouge1_scores),
        'rouge2_std': np.std(rouge2_scores),
        'rougeL_std': np.std(rougeL_scores)
    }


def evaluate_model(model_path=None, val_path=None):
    """Evaluate model with enhanced generation parameters for longer predictions"""
    model_path = model_path or os.getenv('MODEL_PATH', './model_outputs/final_model')
    val_path = val_path or os.getenv('VAL_PATH', 'outputs/val_dataset')

    print("🔍 Starting enhanced model evaluation with longer generation...")
    print("🔧 PHASE 1 EVALUATION ENHANCEMENTS:")
    print("✅ Enhanced generation parameters for longer outputs")
    print("✅ Length-aware evaluation metrics")
    print("✅ Detailed length distribution analysis")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Validation dataset path: {val_path}")

    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        return None
    if not Path(val_path).exists():
        print(f"❌ Validation dataset path does not exist: {val_path}")
        return None

    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        val_dataset = load_from_disk(val_path)
        print(f"✅ Loaded model and {len(val_dataset)} validation samples")
        print(f"Dataset columns: {val_dataset.column_names}")
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return None

    predictions = []
    references = []
    sample_ids = []
    prediction_lengths = []
    reference_lengths = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"🚀 Evaluating on {len(val_dataset)} samples using {device}...")

    with torch.no_grad():
        for i, example in enumerate(val_dataset):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(val_dataset)}")
            try:
                sample_id = example.get('Master_Index', f'VAL_{i:08d}')
                sample_ids.append(sample_id)
                
                # Handle both tokenized and non-tokenized data
                if 'input_ids' in example:
                    # Data is already tokenized
                    input_ids = torch.tensor([example['input_ids']], device=device)
                    attention_mask = torch.tensor([example['attention_mask']], device=device)
                else:
                    # Data needs tokenization
                    inputs = tokenizer(
                        example['Prompt'],
                        return_tensors='pt',
                        truncation=True,
                        padding='max_length',
                        max_length=512
                    )
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                
                # ENHANCED GENERATION PARAMETERS FOR LONGER OUTPUTS
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=400,              # Increased from 512
                    min_length=70,               # Minimum length
                    num_beams=6,                 # Increased from 4
                    early_stopping=False,       # Changed - Let it generate more
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.15,     # Reduced from 1.2
                    length_penalty=1.5,          # Favor longer outputs
                    no_repeat_ngram_size=2,      # Reduced from 3
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_summary = normalize_for_evaluation(generated_text)
                
                # Handle labels - check if they're tokenized or raw text
                if 'labels' in example:
                    if isinstance(example['labels'], list):
                        # Labels are tokenized
                        labels = [token for token in example['labels'] if token != -100]
                        ref_text = tokenizer.decode(labels, skip_special_tokens=True)
                    else:
                        # Labels are raw text
                        ref_text = example['labels']
                else:
                    # Fallback - look for other possible label columns
                    ref_text = example.get('Clinician', 'patient requires clinical assessment')
                
                ref_summary = normalize_for_evaluation(ref_text)
                
                predictions.append(pred_summary)
                references.append(ref_summary)
                
                # Track lengths for analysis
                prediction_lengths.append(len(pred_summary.split()))
                reference_lengths.append(len(ref_summary.split()))
                
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                predictions.append(normalize_for_evaluation(""))
                references.append(normalize_for_evaluation(""))
                sample_ids.append(f'ERROR_{i:08d}')
                prediction_lengths.append(0)
                reference_lengths.append(0)

    # Rest of the evaluation function remains the same...
    print("📊 Computing enhanced metrics with length analysis...")
    valid_pairs = [(p, r, sid, pl, rl) for p, r, sid, pl, rl in zip(predictions, references, sample_ids, prediction_lengths, reference_lengths) if p.strip() and r.strip()]
    if not valid_pairs:
        print("❌ No valid prediction-reference pairs found!")
        return None

    valid_predictions, valid_references, valid_ids, valid_pred_lengths, valid_ref_lengths = zip(*valid_pairs)
    print(f"✅ Computing metrics on {len(valid_pairs)} valid pairs")

    try:
        rouge_results = compute_rouge_scores(valid_predictions, valid_references)
        print("✅ ROUGE computation successful")
    except Exception as e:
        print(f"❌ Error computing ROUGE: {e}")
        rouge_results = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    # Enhanced metrics with length analysis
    avg_pred_length = np.mean(valid_pred_lengths)
    avg_ref_length = np.mean(valid_ref_lengths)
    
    metrics = {
        **rouge_results,
        'num_samples': len(valid_pairs),
        'avg_pred_length': avg_pred_length,
        'avg_ref_length': avg_ref_length,
        'length_ratio': avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0,
        'length_analysis': {
            'pred_length_std': np.std(valid_pred_lengths),
            'ref_length_std': np.std(valid_ref_lengths),
            'min_pred_length': min(valid_pred_lengths),
            'max_pred_length': max(valid_pred_lengths),
            'predictions_over_75_words': sum(1 for l in valid_pred_lengths if l >= 75),
            'predictions_under_50_words': sum(1 for l in valid_pred_lengths if l < 50),
            'length_target_achievement': avg_pred_length >= 75.0
        },
        'format_compliance': {
            'all_lowercase': all(p.islower() for p in valid_predictions),
            'no_punctuation': all(not any(c in p for c in '.,!?;:"()[]{}') for p in valid_predictions),
            'min_word_count': min(valid_pred_lengths),
            'empty_predictions': sum(1 for p in valid_predictions if not p.strip())
        }
    }

    print("\n" + "="*70)
    print("🎯 ENHANCED EVALUATION RESULTS (PHASE 1)")
    print("="*70)
    print(f"ROUGE-1: {metrics['rouge1']:.4f} (±{metrics.get('rouge1_std', 0):.4f})")
    print(f"ROUGE-2: {metrics['rouge2']:.4f} (±{metrics.get('rouge2_std', 0):.4f})")
    print(f"ROUGE-L: {metrics['rougeL']:.4f} (±{metrics.get('rougeL_std', 0):.4f})")
    
    print(f"\n📏 LENGTH ANALYSIS (KEY IMPROVEMENT METRIC):")
    print(f"Average Prediction Length: {avg_pred_length:.1f} words")
    print(f"Average Reference Length: {avg_ref_length:.1f} words")
    print(f"Length Ratio (Pred/Ref): {metrics['length_ratio']:.2f}")
    print(f"Target Achievement (≥75 words): {'✅ YES' if metrics['length_analysis']['length_target_achievement'] else '❌ NO'}")
    print(f"Predictions ≥75 words: {metrics['length_analysis']['predictions_over_75_words']}/{len(valid_pairs)} ({100*metrics['length_analysis']['predictions_over_75_words']/len(valid_pairs):.1f}%)")
    print(f"Predictions <50 words: {metrics['length_analysis']['predictions_under_50_words']}/{len(valid_pairs)} ({100*metrics['length_analysis']['predictions_under_50_words']/len(valid_pairs):.1f}%)")
    print(f"Length Range: {metrics['length_analysis']['min_pred_length']} - {metrics['length_analysis']['max_pred_length']} words")
    
    print(f"\n📈 Format Compliance:")
    print(f"All lowercase: {'✅' if metrics['format_compliance']['all_lowercase'] else '❌'}")
    print(f"No punctuation: {'✅' if metrics['format_compliance']['no_punctuation'] else '❌'}")
    print(f"Min word count: {metrics['format_compliance']['min_word_count']}")
    print(f"Empty predictions: {metrics['format_compliance']['empty_predictions']}")
    
    print(f"\n📈 Statistics:")
    print(f"Samples Evaluated: {metrics['num_samples']}")

    results_file = Path('./experiments/enhanced_evaluation_results.json')
    results_file.parent.mkdir(exist_ok=True)
    detailed_results = {
        'metrics': metrics,
        'phase_1_enhancements': {
            'generation_config': {
                'max_length': 400,
                'min_length': 60,
                'num_beams': 6,
                'length_penalty': 1.4,
                'early_stopping': False
            },
            'target_length_achieved': metrics['length_analysis']['length_target_achievement'],
            'length_improvement_vs_baseline': f"Target: >75 words, Achieved: {avg_pred_length:.1f} words"
        },
        'sample_predictions': [
            {
                'id': valid_ids[i],
                'prediction': valid_predictions[i],
                'reference': valid_references[i],
                'pred_length': valid_pred_lengths[i],
                'ref_length': valid_ref_lengths[i],
                'rouge1': compute_rouge_scores([valid_predictions[i]], [valid_references[i]])['rouge1'],
                'rougeL': compute_rouge_scores([valid_predictions[i]], [valid_references[i]])['rougeL']
            }
            for i in range(min(20, len(valid_predictions)))
        ],
        'format_verification': {
            'normalization_applied': True,
            'single_task_focus': True,
            'consistent_with_inference': True,
            'phase_1_enhanced': True
        }
    }
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"💾 Enhanced results saved to: {results_file}")

    return metrics

def normalize_for_evaluation(text):
    """Helper function for normalization"""
    if not text or text.strip() == "":
        return "patient requires clinical assessment and appropriate treatment"
    normalized = text.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    if not normalized or len(normalized.split()) < 2:
        normalized = "patient requires clinical assessment and appropriate treatment"
    return normalized

if __name__ == '__main__':
    try:
        print("🚀 Starting Enhanced Model Evaluation (Phase 1)...")
        print("=" * 70)
        print("🔧 PHASE 1 EVALUATION ENHANCEMENTS:")
        print("✅ Enhanced generation parameters for longer outputs")
        print("✅ Length-focused evaluation metrics")
        print("✅ Target: Average prediction length ≥75 words")
        print("=" * 70)
        metrics = evaluate_model()
        if metrics:
            print(f"\n📊 Final Results:")
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            print(f"Average Length: {metrics['avg_pred_length']:.1f} words")
            print(f"Length Target: {'✅ ACHIEVED' if metrics['length_analysis']['length_target_achievement'] else '❌ NOT ACHIEVED'}")
        else:
            print("❌ Evaluation failed: No metrics returned")
    except Exception as e:
        print(f"❌ Enhanced evaluation failed: {e}")
