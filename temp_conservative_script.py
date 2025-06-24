
import sys
sys.path.append('scripts')

# Import our conservative inference
exec(open('scripts/conservative_inference.py').read())

# Run conservative inference
try:
    submission_path = run_conservative_inference(
        model_path='experiments/optimized_v2/final_model',
        test_path='outputs/test_dataset',
        output_path='outputs/submission_conservative.csv'
    )
    print(f"✅ Conservative inference completed! Submission: {submission_path}")
except Exception as e:
    print(f"❌ Conservative inference failed: {e}")
    import traceback
    traceback.print_exc()
