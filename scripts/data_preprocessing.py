import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import re
import numpy as np
import os
import torch
from typing import List, Dict
import random

def load_and_normalize_data(train_path, test_path):
    """Load and normalize data with simplified prompt format"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Normalize text
    for df in [train_df, test_df]:
        df['Prompt'] = df['Prompt'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace('\n', ' ')
        if 'Clinician' in df.columns:
            df['Clinician'] = df['Clinician'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace('\n', ' ')

    # Simplified prompt format: "Clinical scenario: {prompt}"
    for df in [train_df, test_df]:
        df['Prompt'] = df.apply(lambda row: f"Clinical scenario: {row['Prompt']}", axis=1)

    print("✅ Applied simplified prompt format")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

def apply_synonym_replacement(text: str) -> str:
    """Apply synonym replacement for data augmentation"""
    medical_synonyms = {
        'patient': ['individual', 'person'],
        'presents': ['shows', 'exhibits'],
        'symptoms': ['signs', 'manifestations'],
        'condition': ['state', 'situation'],
        'treatment': ['care', 'management']
    }

    words = text.split()
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?')
        if word_lower in medical_synonyms and random.random() < 0.2:
            replacement = random.choice(medical_synonyms[word_lower])
            words[i] = word.replace(word_lower, replacement)

    return ' '.join(words)

def apply_noise_injection(text: str) -> str:
    """Inject noise by randomly swapping adjacent words"""
    words = text.split()
    if len(words) > 1 and random.random() < 0.1:  # 10% chance to swap
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return ' '.join(words)

def augment_dataset(dataset, augmentation_factor=1):
    augmentation_factor = dataset[0].get('config', {}).get('data', {}).get('augmentation', {}).get('augmentation_factor', augmentation_factor)
    """Apply basic augmentation: synonym replacement and noise injection"""
    print(f"🔄 Applying basic augmentation with factor {augmentation_factor}...")

    original_data = [example for example in dataset]
    augmented_data = []

    for example in original_data:
        for _ in range(augmentation_factor):
            augmented_prompt = apply_synonym_replacement(example['Prompt'])
            augmented_prompt = apply_noise_injection(augmented_prompt)
            augmented_example = example.copy()
            augmented_example['Prompt'] = augmented_prompt
            augmented_data.append(augmented_example)

    combined_data = original_data + augmented_data
    combined_dataset = Dataset.from_list(combined_data)
    print(f"✅ Augmentation complete! Dataset size: {len(original_data)} → {len(combined_dataset)}")
    return combined_dataset

def tokenize_data(dataset, tokenizer_name='t5-small', max_length=512, has_labels=True):
    """Tokenize data using the default t5-small tokenizer without special tokens"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if has_labels:
        def tokenize(example):
            inputs = tokenizer(
                example['Prompt'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors=None
            )
            if 'Clinician' in example and example['Clinician'] is not None:
                labels = tokenizer(
                    example['Clinician'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors=None
                )
                inputs['labels'] = labels['input_ids']
            return inputs

        return dataset.map(tokenize, batched=True)
    else:
        def tokenize(example):
            inputs = tokenizer(
                example['Prompt'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors=None
            )
            return inputs

        return dataset.map(tokenize, batched=True)

def split_train_val(dataset, val_split=0.2):
    """Split training and validation data"""
    split = dataset.train_test_split(test_size=val_split, seed=42)
    return split['train'], split['test']

if __name__ == "__main__":
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for train data at: {train_path}")
    print(f"Looking for test data at: {test_path}")
    print(f"Train file exists: {os.path.exists(train_path)}")
    print(f"Test file exists: {os.path.exists(test_path)}")

    print("\n🔧 PRIORITY FIXES APPLIED:")
    print("=" * 60)
    print("✅ Removed multitask learning components")
    print("✅ Simplified prompt format")
    print("✅ Removed few-shot examples")
    print("✅ Implemented basic augmentation (synonym replacement and noise injection)")
    print("✅ Consistent tokenizer handling with default t5-small")
    print("=" * 60)

    # Load and normalize data
    print("\n📊 Loading and normalizing data with simplified format...")
    train_dataset, test_dataset = load_and_normalize_data(train_path, test_path)

    # Apply basic augmentation
    print("\n🔄 Applying basic augmentation (1x)...")
    train_dataset = augment_dataset(train_dataset, augmentation_factor=1)

    # Tokenize datasets with default tokenizer
    print("\n🔤 Tokenizing datasets with default t5-small tokenizer...")
    train_dataset = tokenize_data(train_dataset, max_length=512, has_labels=True)
    test_dataset = tokenize_data(test_dataset, max_length=512, has_labels=False)

    # Split training data
    print("\n📊 Splitting training data...")
    train_dataset, val_dataset = split_train_val(train_dataset, val_split=0.15)

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Save datasets
    print("\n💾 Saving simplified datasets...")
    train_dataset.save_to_disk('outputs/train_dataset')
    val_dataset.save_to_disk('outputs/val_dataset')
    test_dataset.save_to_disk('outputs/test_dataset')

    print("\n✅ PRIORITY PREPROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📊 Final dataset sizes:")
    print(f"   Train: {len(train_dataset)} (with basic augmentation)")
    print(f"   Validation: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")

    # Verify a sample to ensure format is correct
    print(f"\n🔍 SAMPLE VERIFICATION:")
    print("-" * 40)
    sample = train_dataset[0]
    print(f"Sample prompt: {sample['Prompt'][:100]}...")
    if 'Clinician' in sample:
        print(f"Sample target: {sample['Clinician'][:100]}...")
    print("=" * 60)