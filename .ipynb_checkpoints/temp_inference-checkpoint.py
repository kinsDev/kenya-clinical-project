
import sys
sys.path.append('scripts')
from inference import run_inference

# Run inference with the best model
try:
    submission_path = run_inference(
        model_path='experiments/baseline/optimized_model/fp16',
        test_path='outputs/test_dataset',
        output_path='outputs/submission.csv',
        use_optimized=False  # We're already using the optimized model
    )
    print(f"✅ Inference completed! Submission saved to: {submission_path}")
except Exception as e:
    print(f"❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
