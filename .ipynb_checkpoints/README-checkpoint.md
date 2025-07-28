# Kenya Clinical Healthcare Project

This project develops an AI model to predict clinician responses to medical scenarios in rural Kenyan healthcare settings, replicating the expertise of trained professionals. It aims to support healthcare workers in resource-limited environments by providing accurate, context-aware responses. The model, based on T5, is evaluated using ROUGE scores, which measure how closely its predictions match expert responses. The project is designed to be modular, scalable, and user-friendly, with a clear directory structure, modular scripts, and a Jupyter notebook for orchestration.

## Directory Structure

```plaintext
user-default-efs/
└── kenya-healthcare-project/
    ├── data/
    │   ├── train.csv
    │   └── test.csv
    │
    ├── experiments/                      # Stores trained models based on config file names
    │
    ├── scripts/
    │   ├── data_preprocessing.py
    │   ├── model_training.py
    │   ├── model_evaluation.py
    │   ├── model_optimization.py
    │   ├── inference.py
    │   ├── utils.py
    │   └── run_experiments.py
    │
    ├── conf/
    │   ├── config.yaml
    │   └── experiments/
    │       ├── baseline.yaml
    │       ├── conservative.yaml
    │       ├── fast.yaml
    │       ├── heavy_reg.yaml
    │       ├── high_lr.yaml
    │       ├── large_batch.yaml
    │       ├── quality.yaml
    │       ├── enhanced.yaml
    │       ├── baseline_v2.yaml
    │       ├── optimized_v2.yaml
    │       ├── optimized_adaptive.yaml
    │       ├── baseline_adaptive.yaml
    │       ├── optimized_enhanced.yaml
    │       └── baseline_enhanced.yaml
    │
    ├── outputs/
    │   ├── train_dataset/
    │   ├── val_dataset/
    │   ├── test_dataset/
    │   └── submission.csv
    │
    ├── model_pipeline.ipynb
    ├── requirements.txt
    └── README.md
```

## File and Directory Functions

### data/
- **train.csv**: Contains 400 training samples with columns:
  - `Master_Index`: Unique identifier for each scenario.
  - `County`: Kenyan county (e.g., Kiambu, Kakamega).
  - `Health level`: Facility type (e.g., Sub-county Hospitals, Dispensaries).
  - `Prompt`: Clinical scenario with nurse background and patient case.
  - `Nursing Competency`: Nursing specialty (e.g., Maternal and Child Health).
  - `Clinical Panel`: Medical specialty for evaluation (e.g., OBSTETRICS).
  - `Clinician`: Expert response (target variable).
  - Other columns: Model responses (e.g., GPT4.0, LLAMA) and SNOMED codes.
- **test.csv**: Contains 100 test samples with similar columns, excluding `Clinician`.

### experiments/
- Stores trained models, checkpoints, and logs for each experiment, named after the configuration file (e.g., `experiments/length_optimized/`).

### scripts/
- **data_preprocessing.py**: Loads, cleans, and augments data from `train.csv` and `test.csv`. Saves datasets in Hugging Face format to `outputs/`.
- **model_training.py**: Trains a T5 model using configurations from `conf/experiments/`. Supports early stopping, Adafactor optimizer, and real-time monitoring.
- **model_evaluation.py**: Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) on the validation dataset.
- **model_optimization.py**: Implements hyperparameter tuning and optimization strategies.
- **inference.py**: Generates predictions for `test.csv`, saving results to `outputs/submission.csv`.
- **utils.py**: Provides shared functions for logging, file handling, and configuration management.
- **run_experiments.py**: Orchestrates multiple experiments in batches, supporting parallel training of up to two models.

### conf/
- **config.yaml**: Base configuration with default settings.
- **experiments/**: YAML files specifying hyperparameters for experiments (e.g., `length_optimized.yaml`).

### outputs/
- **train_dataset/**: Preprocessed training dataset (1700 samples).
- **val_dataset/**: Preprocessed validation dataset (300 samples).
- **test_dataset/**: Preprocessed test dataset (100 samples).
- **submission.csv**: Predictions in the format `Master_Index`, `Clinician`.

### model_pipeline.ipynb
- Orchestrates the pipeline via cells:
  - **Cell 1**: Installs dependencies.
  - **Cell 2**: Runs `data_preprocessing.py`.
  - **Cell 3**: Executes `run_experiments.py` for training.
  - **Cell 4**: Runs `inference.py` for predictions.

### requirements.txt
- Lists dependencies (e.g., `transformers`, `datasets`, `hydra-core`).

### README.md
- This file, providing a comprehensive project overview.

## Project Overview

### Objective
The project aims to predict clinician responses to clinical scenarios in rural Kenyan healthcare settings, considering resource constraints. It uses a T5-based model to generate responses that mimic expert reasoning, evaluated using ROUGE scores.

### Dataset
- **Training Set**: 400 samples, augmented to 1700 training and 300 validation samples.
- **Test Set**: 100 samples for prediction.
- **Key Columns**:
  - `Prompt`: Nurse background and patient scenario.
  - `Clinician`: Target response, normalized for ROUGE (lowercase, no punctuation, single spaces).
  - `County` and `Health level`: Contextualize resource availability.
  - `DDX SNOMED`: Standardized diagnostic codes.

### Evaluation
- **Metric**: ROUGE score (ROUGE-1, ROUGE-2, ROUGE-L).
- **Latest Results** (length_optimized experiment):
  - ROUGE-1: 0.4540 (±0.1417)
  - ROUGE-2: 0.2916 (±0.1683)
  - ROUGE-L: 0.3908 (±0.1558, validation), 0.3896 (leaderboard)
- **Length Analysis**:
  - Average prediction length: 90.2 words
  - Average reference length: 110.0 words
  - Length ratio: 0.82
  - Predictions ≥75 words: 70.6%

## Data Preprocessing

The `data_preprocessing.py` script is crucial for preparing the dataset, addressing the challenge of a small dataset (400 training samples). It performs:

- **Loading and Cleaning**:
  - Loads `train.csv` and `test.csv`.
  - Normalizes `Clinician` responses (lowercase, removes punctuation, single spaces) for ROUGE compatibility.
  - Fixes typos (e.g., "silver sulpha fizika" → "silver sulfadiazine").

- **Data Augmentation**:
  - Expands the dataset to 1700 training and 300 validation samples (total ~2000 samples, with 1496 original and 204 augmented prompts).
  - Techniques include:
    - **Synonym Replacement**: Replaces terms (e.g., "nurse" → "healthcare worker", "fever" → "pyrexia") using dictionaries like `MEDICAL_SYNONYMS` and `NON_MEDICAL_SYNONYMS`.
    - **Question Variation**: Rephrases questions (e.g., "What’s the first step in managing this?" → "What immediate care is needed?").
    - **Sentence Restructuring**: Alters sentence structure while preserving meaning.
    - **Scenario Expansion**: Adds context (e.g., "in the morning", "at the village clinic").
    - **Colloquialisms**: Sparingly uses Kenyan terms (e.g., "mild pain" → "kidogo pain").
    - **Demographic Variation**: Adjusts non-critical details (e.g., "three-year-old child" → "four-year-old boy").
    - **Controlled Noise**: Introduces minor changes (e.g., word swaps) while protecting medical terms.
    - **Typos**: Adds realistic typos to non-medical terms (e.g., "experience" → "experiance").
  - Validation ensures augmented prompts are sufficiently different and preserve medical terms.

- **Example**:
  - **Original Prompt**: "I am a nurse with 22 years of experience in general nursing working in a clinic in Kiambu County in Kenya. A three-year-old child was brought to the facility..."
  - **Augmented Prompt**: "Clinical scenario: I am a healthcare worker with 22 years in practice in general nursing working in a health centre in Kiambu County in Kenya. A three-year-old minor was brought to the facility..."
  - **Sample Stats**:
    - Prompt length: ~770 characters
    - Original prompts: 1496
    - Augmented prompts: 204

- **Output**:
  - **Train size**: 1700 samples
  - **Validation size**: 300 samples
  - **Test size**: 100 samples
  - Saved in Hugging Face format to `outputs/train_dataset/`, `outputs/val_dataset/`, and `outputs/test_dataset/`.
  - Logs: `outputs/preprocessing_log.txt`, `outputs/preprocessing_debug.txt`.

This augmentation increases dataset diversity, helping the model learn robust patterns while maintaining clinical accuracy.

## Experiment Configurations and Workflow

### Configuration Management with Hydra
The project uses [Hydra](https://hydra.cc/), a framework for managing configurations, to streamline experiment setup:
- **Configuration Files**: Each experiment has a YAML file in `conf/experiments/` (e.g., `length_optimized.yaml`) specifying hyperparameters like learning rate, batch size, epochs, and generation settings.
- **Hydra Integration**: In `model_training.py`, Hydra loads configurations dynamically:
  ```python
  @hydra.main(version_base=None, config_path="../conf/experiments", config_name="length_optimized")
  def main(cfg: DictConfig):
      ...
  ```
  This allows seamless switching between experiments by changing the config name.
- **utils.py**: Provides functions for configuration validation and logging, ensuring consistency across scripts.

### Experiment Runner
- **run_experiments.py**: Manages experiment batches, supporting parallel training of up to two models using `ThreadPoolExecutor`. For example:
  - Batch 6 runs `length_optimized`.
  - Batch 1 runs `baseline` and `quality` simultaneously.
- **Process**:
  - Loads configuration (e.g., `length_optimized.yaml`).
  - Runs `model_training.py` for training.
  - Executes `model_evaluation.py` for ROUGE scores.
  - Saves results to `experiments/experiment_results.json`.

### Model Training
- **model_training.py**: Uses T5-base with the Adafactor optimizer. Key features:
  - Early stopping based on evaluation loss.
  - Real-time monitoring of loss and learning rate.
  - Saves models to `experiments/<config_name>/final_model`.

### Connection Between Components
- **utils.py**: Provides shared functions (e.g., logging, file handling) used by all scripts.
- **Notebook Integration**: `model_pipeline.ipynb` orchestrates the pipeline by calling scripts and leveraging configurations, ensuring a cohesive workflow.
- **Scalability**: The modular design allows new experiments to be added by creating new YAML files, and the pipeline can be automated for production.

### Parallel Training
- The project trains up to two models simultaneously, speeding up experimentation. For example, in Batch 1, `baseline` and `quality` models are trained in parallel, reducing total runtime.

## Jupyter Notebook and Scalable Pipeline

The `model_pipeline.ipynb` notebook is the central interface:
- **Cell 1**: Installs dependencies.
- **Cell 2**: Runs `data_preprocessing.py` to prepare datasets.
- **Cell 3**: Executes `run_experiments.py` for training and monitoring.
- **Cell 4**: Runs `inference.py` to generate predictions.

This design ensures scalability:
- **Modularity**: Scripts are independent, allowing easy updates or additions.
- **Configurability**: Hydra enables flexible hyperparameter tuning.
- **Production Readiness**: The pipeline can be automated or deployed on cloud platforms, making it suitable for real-world healthcare applications.

## Performance Analysis

### Recent Results (Batch 6: length_optimized)
- **Training Time**: 35.2 minutes
- **ROUGE Scores**:
  - ROUGE-1: 0.4540 (±0.1417)
  - ROUGE-2: 0.2916 (±0.1683)
  - ROUGE-L: 0.3908 (±0.1558, validation), 0.3896 (leaderboard)
- **Length Analysis**:
  - Average prediction length: 90.2 words
  - Average reference length: 110.0 words
  - Length ratio: 0.82
  - Predictions ≥75 words: 70.6%
- **Observations**:
  - Training loss stabilizes, but evaluation loss increases after ~600 steps, indicating overfitting.
  - Learning rate peaks at 0.001 (step 100) and decays to 0.0004 (step 1000).
  - Shorter predictions may reduce ROUGE scores.

### Recommendations
- **Reduce Overfitting**:
  - Set `early_stopping_patience: 5` to stop at ~800 steps.
  - Increase `weight_decay` to 0.02–0.05.
- **Optimize Inference**:
  - Set `length_penalty: 1.0` to encourage longer sequences.
  - Increase `num_beams` to 8.
- **Model Size**: Test T5-small to reduce overfitting.
- **Learning Rate**: Use a linear decay scheduler.

## Setup and Execution

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Pipeline
1. **Preprocessing**:
   ```bash
   python scripts/data_preprocessing.py
   ```
2. **Training**:
   ```bash
   python scripts/model_training.py --config-path=conf/experiments --config-name=length_optimized
   ```
3. **Evaluation**:
   ```bash
   python scripts/model_evaluation.py
   ```
4. **Inference**:
   ```bash
   python scripts/inference.py
   ```
5. **Experiments**:
   ```bash
   python scripts/run_experiments.py 6
   ```

### Using the Notebook
- Open `model_pipeline.ipynb` in Jupyter.
- Execute cells sequentially.

## Experiment Batches

| Batch | Configurations |
|-------|----------------|
| 1     | baseline, quality |
| 2     | enhanced, optimized |
| 3     | baseline_v2, optimized_v2 |
| 4     | optimized_adaptive, baseline_adaptive |
| 5     | optimized_enhanced, baseline_enhanced |
| 6     | length_optimized |

## Layman/Business Terms

- **What is this project?**
  - Imagine an AI that acts like a doctor or nurse, giving advice for medical situations in rural Kenya where experts are scarce. This project trains a computer to provide these answers accurately.

- **Why is it important?**
  - In rural areas, there aren’t always enough healthcare workers. This AI can help by suggesting what a clinician would do, potentially improving patient care and saving lives.

- **How does it work?**
  - We start with 400 real medical scenarios and expert responses. The AI learns from these, using techniques to create more examples (like rephrasing sentences). It then predicts answers for new scenarios, and we check how close it gets to the experts’ responses.

- **How is it built?**
  - A Jupyter notebook runs scripts to prepare data, train models, and test them. We can tweak settings (like how fast the AI learns) using configuration files, and we can train two models at once to compare them quickly.

- **Can it be used in the real world?**
  - Yes! The system is designed to be flexible and can be scaled up to work in hospitals or on cloud platforms, making it practical for real healthcare needs.

## Contact
- X: @king_sley007