import random
import pandas as pd
from datasets import Dataset
from typing import List, Dict
import logging
import os
import re
from nltk import sent_tokenize
from transformers import T5Tokenizer

# Set up logging
logging.basicConfig(filename='outputs/preprocessing_debug.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Expanded medical synonym dictionary with symptom terms
MEDICAL_SYNONYMS = {
    "malaria": ["plasmodium infection", "fever disease", "malarial illness"],
    "tuberculosis": ["TB", "pulmonary TB", "consumption"],
    "fever": ["high temperature", "pyrexia", "hotness"],
    "pain": ["ache", "discomfort", "soreness"],
    "vomiting": ["emesis", "throwing up", "nausea"],
    "headache": ["head pain", "migraine", "cephalalgia"],
    "clinic": ["health centre", "dispensary", "outpatient facility"],
    "hospital": ["medical facility", "health institution", "ward"],
    "diarrhea": ["loose stools", "diarrhoea", "stomach upset"],
    "cough": ["chest infection", "coughing", "respiratory distress"],
    "injury": ["wound", "trauma", "hurt"],
    "matatu accident": ["road crash", "vehicle collision", "traffic accident"],
    "treatment": ["care", "therapy", "management"],
    "patient": ["client", "case", "individual"],
    "doctor": ["physician", "clinician", "medical officer"],
    "burn": ["scald", "thermal injury", "skin burn"],
    "swelling": ["edema", "puffiness", "inflammation"],
    "fatigue": ["tiredness", "exhaustion", "weariness"],
    "nausea": ["sickness", "queasiness", "stomach upset"],
    "chills": ["shivers", "cold spells", "shaking"],
    "dizziness": ["lightheadedness", "vertigo", "unsteadiness"],
    "rash": ["skin eruption", "hives", "red spots"],
    "bleeding": ["hemorrhage", "blood loss", "leakage"],
    "weakness": ["lethargy", "feebleness", "low energy"],
}

# Non-medical synonyms (clinical tone preserved)
NON_MEDICAL_SYNONYMS = {
    "years of experience": ["years in practice", "years working"],
    "child": ["young patient", "minor"],
    "nurse": ["healthcare worker", "medical staff"],
    "staff": ["team", "personnel"],
    "health": ["wellness", "condition"],
}

# Question templates with partial matching keys
QUESTION_TEMPLATES = {
    r"immediate.*treatment|urgent.*care|first.*step": [
        "How should this be treated right away?",
        "What’s the first step in managing this?",
        "What immediate care is needed?"
    ],
    r"tetanus.*prophylaxis|prevention.*tetanus": [
        "Is tetanus prevention necessary here?",
        "Does this case require tetanus prophylaxis?"
    ],
    r"follow.*up|aftercare|post.*care": [
        "What aftercare is advised?",
        "What should be done for follow-up?"
    ],
    r"symptoms|what.*signs": [
        "What are the key symptoms here?",
        "What signs should I look for?"
    ],
    r"test|investigation|examination": [
        "What tests are needed for this?",
        "Which investigations should be done?"
    ],
    r"prognosis|outcome|future": [
        "What’s the likely prognosis?",
        "How will this case progress?"
    ],
}

# Detail templates for scenario expansion
DETAIL_TEMPLATES = {
    "time": ["in the morning", "at night", "during the day", "in the afternoon", "early evening", "during a meal"],
    "location": ["at the village clinic", "in the outpatient department", "at the rural health post", "in the emergency room", "at a local dispensary"],
    "pediatric": ["after school", "during playtime", "after a nap", "during a family gathering", "at bedtime"],
}

# Kenyan colloquialisms (used sparingly)
COLLOQUIALISMS = {
    "mild pain": "kidogo pain",
    "slight fever": "joto kidogo",
    "small pain": "pain small small",
    "body hot": "hot body",
    "feeling bad": "sick sick",
}

# Common typos for non-medical terms
TYPOS = {
    "experience": "experiance",
    "working": "workin",
    "clinic": "clinc",
    "nurse": "nuse",
    "patient": "paitent",
    "staff": "staf",
    "health": "helth",
}

# Do-not-alter list for critical terms
DO_NOT_ALTER = list(MEDICAL_SYNONYMS.keys())

def clean_clinician_text(text: str) -> str:
    """
    Cleans Clinician column text by fixing common typos and standardizing format.
    """
    text = str(text)
    text = re.sub(r'\bto to\b', 'to', text)
    text = re.sub(r'\bsilver sulpha fizika\b', 'silver sulfadiazine', text, flags=re.IGNORECASE)
    text = re.sub(r'\buncompliance\b', 'noncompliance', text, flags=re.IGNORECASE)
    text = re.sub(r'\bknon\b', 'known', text, flags=re.IGNORECASE)
    if 'ID_QOQTK' in text:
        text = re.sub(r'\b72 year old\b', '92 year old', text)
    sections = ['summary', 'diagnosis', 'management', 'investigations']
    for section in sections:
        if section in text.lower():
            text = text.replace(section, f'\n- {section.capitalize()}:')
    return text.strip()

def normalize_text(text: str) -> str:
    """Normalize text for ROUGE compatibility."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces/newlines with single space
    return text.strip()
    
def is_valid_paraphrase(original: str, paraphrase: str) -> bool:
    """Validate if paraphrase is sufficiently different and preserves medical terms."""
    if not paraphrase or len(paraphrase.split()) < 5:
        logging.debug(f"Paraphrase rejected: too short or empty - {paraphrase[:200]}...")
        return False
    original_words = set(original.lower().split())
    paraphrase_words = set(paraphrase.lower().split())
    common_words = original_words.intersection(paraphrase_words)
    similarity = len(common_words) / max(len(original_words), 1)
    if similarity >= 0.95:
        logging.debug(f"Paraphrase rejected: too similar (similarity={similarity:.2f}) - {paraphrase[:200]}...")
        return False
    if len(paraphrase_words) < len(original_words) * 0.4:
        logging.debug(f"Paraphrase rejected: too few words ({len(paraphrase_words)} vs {len(original_words)}*0.4) - {paraphrase[:200]}...")
        return False
    medical_terms = set(MEDICAL_SYNONYMS.keys())
    original_medical = medical_terms.intersection(original_words)
    paraphrase_medical = medical_terms.intersection(paraphrase_words)
    if original_medical != paraphrase_medical:
        logging.debug(f"Paraphrase rejected: missing medical terms {original_medical - paraphrase_medical}")
        return False
    return True

def apply_contextual_augmentation(text: str) -> str:
    """
    Replaces non-medical terms only.
    """
    for key, synonyms in NON_MEDICAL_SYNONYMS.items():
        if key in text:
            text = text.replace(key, random.choice(synonyms))
    return text

def apply_question_variation(prompt: str) -> str:
    """
    Applies question variation using regex for partial matching.
    """
    for pattern, variants in QUESTION_TEMPLATES.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            return re.sub(pattern, random.choice(variants), prompt, flags=re.IGNORECASE)
    return prompt

def apply_synonym_replacement(text: str) -> str:
    """
    Replaces medical and non-medical terms with synonyms.
    """
    words = text.split()
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?')
        if word_lower in MEDICAL_SYNONYMS and random.random() < 0.9:
            replacement = random.choice(MEDICAL_SYNONYMS[word_lower])
            words[i] = replacement if word.islower() else replacement.capitalize()
        elif word_lower in NON_MEDICAL_SYNONYMS and random.random() < 0.5:
            replacement = random.choice(NON_MEDICAL_SYNONYMS[word_lower])
            words[i] = replacement if word.islower() else replacement.capitalize()
    return ' '.join(words)

def apply_sentence_restructuring(text: str) -> str:
    """
    Restructures sentences with multiple patterns.
    """
    sentences = sent_tokenize(text)
    restructured = []
    for sentence in sentences:
        if "presents with" in sentence:
            parts = sentence.split("presents with")
            restructured.append(f"{parts[0].strip()} has {parts[1].strip()}")
        elif re.search(r"(\w+)\s+touched\s+(\w+)", sentence):
            match = re.search(r"(\w+)\s+touched\s+(\w+)", sentence)
            restructured.append(f"{match.group(2)} was touched by {match.group(1)}")
        elif "," in sentence and "who" in sentence:
            parts = sentence.split(",", 1)
            restructured.append(f"{parts[1].strip()} {parts[0].strip()}")
        else:
            restructured.append(sentence)
    return ' '.join(restructured)

def apply_controlled_noise(text: str) -> str:
    """
    Introduces noise while protecting medical terms.
    """
    words = text.split()
    if len(words) > 5:
        for _ in range(int(len(words) * 0.05)):
            idx = random.randint(0, len(words) - 2)
            if words[idx].lower() not in DO_NOT_ALTER and words[idx + 1].lower() not in DO_NOT_ALTER:
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        if random.random() < 0.1:
            idx = random.randint(0, len(words) - 1)
            if words[idx].lower() not in DO_NOT_ALTER:
                del words[idx]
    return ' '.join(words)

def apply_scenario_expansion(prompt: str) -> str:
    """
    Adds context-aware scenario details.
    """
    if "child" in prompt.lower():
        if "school" not in prompt.lower():
            return prompt + " " + random.choice(DETAIL_TEMPLATES["pediatric"])
    if not any(t in prompt.lower() for t in ["morning", "night", "day", "afternoon", "evening", "meal"]):
        return prompt + " " + random.choice(DETAIL_TEMPLATES["time"])
    if not any(l in prompt.lower() for l in ["clinic", "hospital", "health post", "emergency room", "dispensary"]):
        return prompt + " " + random.choice(DETAIL_TEMPLATES["location"])
    return prompt

def apply_prompt_truncation(prompt: str) -> str:
    """
    Removes non-essential intro with 10% probability.
    """
    if random.random() < 0.1:
        sentences = sent_tokenize(prompt)
        if len(sentences) > 1 and "I am a nurse" in sentences[0]:
            return ' '.join(sentences[1:])
    return prompt

def apply_colloquialisms(text: str) -> str:
    """
    Adds Kenyan colloquialisms sparingly (10% probability).
    """
    if random.random() < 0.1:
        for key, value in COLLOQUIALISMS.items():
            if key in text:
                text = text.replace(key, value)
    return text

def apply_demographic_variation(text: str) -> str:
    """
    Varies non-critical demographic details (e.g., age, gender).
    """
    if "year-old child" in text:
        age = random.randint(3, 6)
        gender = random.choice(["boy", "girl"])
        text = re.sub(r'\d+-year-old child', f"{age}-year-old {gender}", text)
    return text

def apply_typos(text: str) -> str:
    """
    Adds typos to non-medical terms with 5% probability.
    """
    words = text.split()
    for i, word in enumerate(words):
        if word in TYPOS and word.lower() not in DO_NOT_ALTER and random.random() < 0.05:
            words[i] = TYPOS[word]
    return ' '.join(words)

def augment_prompt(prompt: str, augmentation_factor: int = 3) -> List[Dict]:
    """
    Augments prompt with a random subset of techniques.
    """
    augmented_data = [{"Prompt": prompt, "augmentation_type": "original"}]
    attempts = 0
    max_attempts = augmentation_factor * 5
    techniques = [
        apply_contextual_augmentation,
        apply_question_variation,
        apply_synonym_replacement,
        apply_sentence_restructuring,
        apply_controlled_noise,
        apply_scenario_expansion,
        apply_prompt_truncation,
        apply_colloquialisms,
        apply_demographic_variation,
        apply_typos,
    ]

    while len(augmented_data) < augmentation_factor and attempts < max_attempts:
        selected = random.sample(techniques, k=random.randint(2, 4))
        new_prompt = prompt
        applied = []
        for func in selected:
            new_prompt = func(new_prompt)
            applied.append(func.__name__)
        if is_valid_paraphrase(prompt, new_prompt) and new_prompt not in [d["Prompt"] for d in augmented_data]:
            augmented_data.append({"Prompt": new_prompt, "augmentation_type": "augmented", "techniques": applied})
            similarity = len(set(new_prompt.lower().split()).intersection(set(prompt.lower().split())))/max(len(set(prompt.lower().split())),1)
            logging.info(f"Generated valid prompt (similarity={similarity:.2f}, techniques={applied}): {new_prompt[:200]}...")
        attempts += 1

    while len(augmented_data) < augmentation_factor:
        new_prompt = apply_synonym_replacement(prompt)
        if is_valid_paraphrase(prompt, new_prompt):
            augmented_data.append({"Prompt": new_prompt, "augmentation_type": "augmented", "techniques": ["apply_synonym_replacement"]})
        else:
            augmented_data.append({"Prompt": prompt, "augmentation_type": "original"})
    return augmented_data

def preprocess_data(input_file: str, output_dir: str, augmentation_factor: int = 3) -> None:
    """Preprocess and augment dataset, saving as Hugging Face datasets."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    if 'Prompt' not in df.columns or 'Clinician' not in df.columns:
        raise ValueError("Input CSV must contain 'Prompt' and 'Clinician' columns")
    logging.info(f"Loaded {input_file} with {len(df)} rows")

    # Clean and normalize Clinician for ROUGE compatibility
    df['Clinician'] = df['Clinician'].apply(clean_clinician_text)
    df['Clinician'] = df['Clinician'].apply(normalize_text)

    # Apply simplified prompt format (inspired by original script)
    df['Prompt'] = df['Prompt'].apply(lambda x: f"Clinical scenario: {x}")

    augmented_data = []
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")  # Updated to t5-base
    for idx, row in df.iterrows():
        prompt = str(row['Prompt'])
        prompt_data = augment_prompt(prompt, augmentation_factor)
        for data in prompt_data:
            new_row = {"Prompt": data["Prompt"], "augmentation_type": data["augmentation_type"]}
            if "techniques" in data:
                new_row["techniques"] = data["techniques"]
            for col in df.columns:
                if col == 'Clinician':
                    new_row['labels'] = str(row[col]) if pd.notnull(row[col]) else ""  # Rename to labels
                elif col != 'Prompt':
                    new_row[col] = str(row[col]) if pd.notnull(row[col]) else ""
            # Check token lengths
            prompt_tokens = len(tokenizer(new_row['Prompt']).input_ids)
            label_tokens = len(tokenizer(new_row['labels']).input_ids) if new_row['labels'] else 0
            if prompt_tokens > 512 or label_tokens > 512:
                logging.warning(f"Row {idx}: Prompt tokens={prompt_tokens}, Label tokens={label_tokens} exceed 512")
            augmented_data.append(new_row)

    dataset = Dataset.from_list(augmented_data)
    train_val_split = dataset.train_test_split(test_size=0.15, seed=42)
    train_val_split['train'].save_to_disk(f"{output_dir}/train_dataset")
    train_val_split['test'].save_to_disk(f"{output_dir}/val_dataset")
    test_df = pd.read_csv("data/test.csv")
    # Apply same prompt format to test data
    test_df['Prompt'] = test_df['Prompt'].apply(lambda x: f"Clinical scenario: {x}")
    Dataset.from_pandas(test_df).save_to_disk(f"{output_dir}/test_dataset")

if __name__ == "__main__":
    preprocess_data("data/train.csv", "outputs", augmentation_factor=3)
