import torch
import torch.quantization as quant
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import os
import time
import gc
import json

def optimize_model(model_path='./model_outputs/final_model', output_path='./model_outputs/optimized_model'):
    print("🔧 Loading model for optimization...")
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Failed to load model/tokenizer: {e}")
        raise
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print("🚀 Applying optimizations...")

    # 1. Dynamic Quantization
    try:
        print("⚡ Applying dynamic quantization...")
        model.eval()
        quantized_model = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        quantized_path = output_path / 'quantized'
        quantized_path.mkdir(exist_ok=True)
        quantized_model.save_pretrained(quantized_path)
        tokenizer.save_pretrained(quantized_path)
        print(f"✅ Quantized model saved to {quantized_path}")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        quantized_model = model

    # 2. Pruning
    try:
        print("✂️ Applying pruning...")
        pruned_model = apply_pruning(model, prune_ratio=0.15)
        pruned_path = output_path / 'pruned'
        pruned_path.mkdir(exist_ok=True)
        pruned_model.save_pretrained(pruned_path)
        tokenizer.save_pretrained(pruned_path)
        print(f"✅ Pruned model saved to {pruned_path}")
    except Exception as e:
        print(f"❌ Pruning failed: {e}")
        pruned_model = model

    # 3. FP16 Conversion
    try:
        print("🎯 Converting to FP16...")
        fp16_model = model.half()
        for module in fp16_model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.half()
        fp16_path = output_path / 'fp16'
        fp16_path.mkdir(exist_ok=True)
        fp16_model.save_pretrained(fp16_path)
        tokenizer.save_pretrained(fp16_path)
        print(f"✅ FP16 model saved to {fp16_path}")
    except Exception as e:
        print(f"❌ FP16 conversion failed: {e}")
        fp16_model = model

    # 4. JIT Compilation
    try:
        print("⚡ Applying JIT compilation...")
        jit_model = apply_jit_optimization(model, tokenizer)
        if jit_model is not None:
            jit_path = output_path / 'jit_compiled.pt'
            torch.jit.save(jit_model, str(jit_path))
            tokenizer.save_pretrained(output_path / 'jit_tokenizer')
            print(f"✅ JIT compiled model saved to {jit_path}")
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")

    # 5. Model Size Comparison
    print("\n" + "="*60)
    print("📊 MODEL SIZE COMPARISON")
    print("="*60)
    original_size = get_model_size(model_path)
    print(f"Original model size: {original_size:.2f} MB")
    for opt_type in ['quantized', 'pruned', 'fp16']:
        opt_path = output_path / opt_type
        if opt_path.exists():
            opt_size = get_model_size(str(opt_path))
            reduction = ((original_size - opt_size) / original_size) * 100
            print(f"{opt_type.capitalize()} model: {opt_size:.2f} MB ({reduction:.1f}% reduction)")
    jit_path = output_path / 'jit_compiled.pt'
    if jit_path.exists():
        jit_size = jit_path.stat().st_size / (1024 * 1024)
        reduction = ((original_size - jit_size) / original_size) * 100
        print(f"JIT compiled model: {jit_size:.2f} MB ({reduction:.1f}% reduction)")

    # 6. Performance Testing
    print("\n" + "="*60)
    print("🚀 PERFORMANCE COMPARISON")
    print("="*60)
    test_enhanced_performance(model_path, output_path, tokenizer)

    # 7. Generate Optimization Report
    optimization_report = {
        'original_model_size_mb': original_size,
        'optimizations': {
            'quantized': {'size_mb': get_model_size(str(output_path / 'quantized'))},
            'pruned': {'size_mb': get_model_size(str(output_path / 'pruned'))},
            'fp16': {'size_mb': get_model_size(str(output_path / 'fp16'))},
            'jit': {'size_mb': jit_path.stat().st_size / (1024 * 1024) if jit_path.exists() else 0.0}
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report_file = output_path / 'optimization_report.json'
    with open(report_file, 'w') as f:
        json.dump(optimization_report, f, indent=2, default=str)
    print(f"\n💾 Optimization report saved to: {report_file}")
    return str(output_path)

def apply_pruning(model, prune_ratio=0.15):
    print(f"Applying pruning with {int(prune_ratio*100)}% weight removal...")
    pruned_model = type(model)(model.config)
    pruned_model.load_state_dict(model.state_dict())
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.data
            if len(weights.shape) == 2:
                row_norms = torch.norm(weights, dim=1)
                col_norms = torch.norm(weights, dim=0)
                threshold = torch.quantile(torch.min(row_norms, col_norms), prune_ratio)
                row_mask = row_norms > threshold
                col_mask = col_norms > threshold
                for i in range(weights.shape[0]):
                    if not row_mask[i]:
                        weights[i, :] = 0
                for j in range(weights.shape[1]):
                    if not col_mask[j]:
                        weights[:, j] = 0
    print("Pruning completed")
    return pruned_model

def apply_jit_optimization(model, tokenizer):
    print("Applying JIT compilation...")
    try:
        model.eval()
        vocab_size = len(tokenizer)
        example_input_ids = torch.randint(0, vocab_size, (1, 128), dtype=torch.long)
        example_attention_mask = torch.ones(1, 128, dtype=torch.long)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (example_input_ids, example_attention_mask), strict=False)
        traced_model = torch.jit.optimize_for_inference(traced_model)
        print("JIT compilation successful")
        return traced_model
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        return None

def test_enhanced_performance(original_path, optimized_path, tokenizer):
    print("🔍 Testing performance...")
    test_inputs = ["Clinical scenario: A 45-year-old patient presents with chest pain and shortness of breath."]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_to_test = {
        'Original': original_path,
        'Quantized': str(optimized_path / 'quantized'),
        'Pruned': str(optimized_path / 'pruned'),
        'FP16': str(optimized_path / 'fp16'),
    }
    for name, path in models_to_test.items():
        if not os.path.exists(path):
            print(f"❌ Model not found: {path}")
            continue
        try:
            model = T5ForConditionalGeneration.from_pretrained(path)
            model.to(device)
            model.eval()
            times = []
            for test_input in test_inputs:
                inputs = tokenizer(test_input, return_tensors='pt', max_length=512, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                run_times = []
                for _ in range(3):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model.generate(**inputs, max_length=100, num_beams=2, do_sample=False)
                    run_times.append(time.time() - start_time)
                times.append(sum(run_times) / len(run_times))
            avg_time = sum(times) / len(times)
            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            print(f"{name}: {avg_time:.3f}s avg, {memory_mb:.1f}MB memory")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"❌ Failed to test {name}: {e}")

def get_model_size(model_path):
    total_size = 0
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path.stat().st_size / (1024 * 1024)
    if not model_path.exists():
        return 0.0
    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)

if __name__ == '__main__':
    try:
        print("🚀 Starting Model Optimization...")
        output_path = optimize_model()
        print(f"\n🎉 Optimization completed! All models saved to: {output_path}")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")