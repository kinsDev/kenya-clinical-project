import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def load_data(file_path: str) -> Dataset:
    """Load data from CSV and convert to Hugging Face Dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} samples from {file_path}")
        return Dataset.from_pandas(df)
    except Exception as e:
        print(f"❌ Failed to load data from {file_path}: {e}")
        raise

def tokenize_text(text: str, tokenizer_name: str, max_length: int = 512) -> dict:
    """Tokenize text using the specified tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer(text, truncation=True, padding='max_length', max_length=max_length)

@hydra.main(
    version_base=None,
    config_path="/mnt/custom-file-systems/efs/fs-0373d59349458e0d3_fsap-00e091e223fc72272/kenya-healthcare-project/conf",
    config_name="config"
)
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print("PRIORITY FIXES: ENHANCED TRAINING PIPELINE")
    print("=" * 60)
    # You can place additional logic here if needed


if __name__ == '__main__':
    main()