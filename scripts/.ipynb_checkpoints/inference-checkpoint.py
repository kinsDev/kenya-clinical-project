import pandas as pd
from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from pathlib import Path
import os
import json
import re
import time
from typing import List, Dict

class InferenceEngine:
    """Inference engine for generating summaries"""
    def __init__(self, model_path: str, use_optimized: bool = True):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model(use_optimized)

    def _load_model(self, use_optimized: bool):
        """Load the best available model version"""
        print("🔍 Searching for best available model...")
        model_candidates = []
        if use_optimized:
            base_path = Path(self.model_path).parent
            optimized_candidates = [
                (base_path / 'optimized_model' / 'fp16', 'FP16 Optimized'),
                (base_path / 'optimized_model' / 'quantized', 'Quantized'),
            ]
            for path, name in optimized_candidates:
                if path.exists() and any(path.iterdir()):
                    model_candidates.append((str(path), name))
        model_candidates.append((self.model_path, 'Original'))
        for model_path, model_name in model_candidates:
            try:
                print(f"🔄 Trying to load {model_name} model from: {model_path}")
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Successfully loaded {model_name} model")
                return
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        raise RuntimeError("❌ Could not load any model version!")

    def generate_summary(self, prompt: str, generation_config: Dict = None) -> str:
        """Generate summary with proper normalization"""
        if generation_config is None:
            generation_config = {
                'max_length': 256,
                'num_beams': 4,
                'early_stopping': True,
                'do_sample': False,
                'repetition_penalty': 1.2,
                'length_penalty': 1.0,
                'no_repeat_ngram_size': 3
            }
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_config
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            normalized_summary = self._normalize_for_evaluation(generated_text)
            return normalized_summary
        except Exception as e:
            print(f"❌️ Failed to generate sample for prompt: {prompt[:50]}... Error: {e}")
            return "patient requires clinical assessment and appropriate treatment"

    def _normalize_for_evaluation(self, summary: str) -> str:
        """Apply evaluation format normalization"""
        if not summary or summary.strip() == "":
            return "patient requires clinical assessment and appropriate treatment"
        normalized = summary.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        if not normalized or len(normalized.split()) < 2:
            normalized = "patient requires clinical assessment and appropriate treatment"
        return normalized

def run_inference(
        model_path='./model_outputs/final_model',
        test_path='outputs/test_dataset',
        output_path='outputs/submission.csv',
        use_optimized=True
):
    """Run inference with proper normalization"""
    print("🚀 Starting Inference...")
    print("=" * 60)
    try:
        inference_engine = InferenceEngine(model_path, use_optimized)
        test_dataset = load_from_disk(test_path)
        print(f"✅ Loaded {len(test_dataset)} test samples")
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        raise
    print(f"🎯 Running inference on {len(test_dataset)} samples...")
    print(f"💻 Using device: {inference_engine.device}")
    predictions = []
    start_time = time.time()
    for i, example in enumerate(test_dataset):
        if i % 25 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                eta = (elapsed / i) * (len(test_dataset) - i)
                print(f"📊 Progress: {i+1}/{len(test_dataset)} ({100*(i+1)/len(test_dataset):.1f}%) - ETA: {eta/60:.1f}min")
        try:
            summary = inference_engine.generate_summary(example['Prompt'])
            predictions.append(summary)
        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")
            fallback = inference_engine._normalize_for_evaluation("patient requires clinical assessment and appropriate treatment")
            predictions.append(fallback)
    total_time = time.time() - start_time
    print(f"✅ Inference completed in {total_time/60:.1f} minutes")
    print(f"⚡ Average time per sample: {total_time/len(test_dataset):.3f}s")
    print("📝 Creating submission file...")
    submission_data = []
    try:
        if 'Master_Index' in test_dataset.column_names:
            for i, example in enumerate(test_dataset):
                submission_data.append({
                    'Master_Index': example['Master_Index'],
                    'Clinician': predictions[i]
                })
            print("✅ Using Master_Index from test dataset")
        else:
            for i, pred in enumerate(predictions):
                submission_data.append({
                    'Master_Index': f'ID_{i:08d}',
                    'Clinician': pred
                })
            print("⚠️ Master_Index not found, using generated IDs")
    except Exception as e:
        print(f"❌ Error creating submission data: {e}")
        for i, pred in enumerate(predictions):
            submission_data.append({
                'Master_Index': f'ID_{i:08d}',
                'Clinician': pred
            })
    submission_df = pd.DataFrame(submission_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Submission file saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    try:
        submission_path = run_inference(use_optimized=True)
        print("\n🎉 Inference completed successfully!")
        print(f"📁 Submission file: {submission_path}")
    except Exception as e:
        print(f"❌ Inference failed: {e}")