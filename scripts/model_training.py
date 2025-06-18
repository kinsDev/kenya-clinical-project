import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from datasets import load_from_disk
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path

class MedicalT5Trainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.validate_config()
        self.model_name = self.config.experiments.model.name
        self.output_dir = self.config.experiments.paths.output_dir
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        except Exception as e:
            print(f"❌ Failed to load tokenizer for {self.model_name}: {e}")
            raise
        self.model = None

    def validate_config(self):
        """Validate configuration to ensure required keys exist"""
        required_keys = [
            ('experiments.model.name', 'Model name (e.g., t5-small)'),
            ('experiments.paths.output_dir', 'Output directory path'),
            ('experiments.vignette_training.epochs', 'Number of training epochs'),
            ('experiments.vignette_training.batch_size', 'Training batch size'),
            ('experiments.vignette_training.eval_batch_size', 'Evaluation batch size'),
            ('experiments.vignette_training.gradient_accumulation_steps', 'Gradient accumulation steps'),
            ('experiments.vignette_training.warmup_steps', 'Warmup steps'),
            ('experiments.vignette_training.weight_decay', 'Weight decay'),
            ('experiments.vignette_training.learning_rate', 'Learning rate'),
            ('experiments.vignette_training.eval_steps', 'Evaluation steps'),
            ('experiments.vignette_training.save_steps', 'Save steps'),
            ('experiments.vignette_training.early_stopping_patience', 'Early stopping patience'),
            ('experiments.vignette_training.label_smoothing_factor', 'Label smoothing factor'),
            ('experiments.paths.logs_dir', 'Logging directory')
        ]
        
        print("🔍 Validating configuration structure...")
        print(f"Config keys available: {list(self.config.keys())}")
        
        for key, desc in required_keys:
            try:
                value = OmegaConf.select(self.config, key)
                if value is None:
                    print(f"❌ Missing config key: {key} ({desc})")
                    print(f"Config structure:\n{OmegaConf.to_yaml(self.config)}")
                    raise ValueError(f"Missing configuration key: {key}")
                else:
                    print(f"✅ Found {key}: {value}")
            except Exception as e:
                print(f"❌ Error accessing config key: {key} ({desc})")
                print(f"Config structure:\n{OmegaConf.to_yaml(self.config)}")
                raise ValueError(f"Configuration error for {key}: {e}")
        print("✅ Configuration validated successfully")

    def load_model(self):
        """Load the standard T5 model"""
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"🔧 Resizing model embeddings: {self.model.config.vocab_size} → {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"✅ Model loaded: {self.model_name}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            return self.model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def train(self, epochs=None):
        """Train the model on vignettes"""
        print("🚀 Starting training...")
        if epochs is None:
            epochs = self.config.experiments.vignette_training.epochs

        try:
            train_dataset = load_from_disk('outputs/train_dataset')
            val_dataset = load_from_disk('outputs/val_dataset')
            print(f"✅ Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        except Exception as e:
            print(f"❌ Failed to load datasets: {e}")
            raise

        if not train_dataset or not val_dataset:
            print("❌ Datasets are empty!")
            raise ValueError("Empty datasets detected")

        sample = train_dataset[0]
        if 'labels' not in sample:
            print("❌ Dataset missing 'labels' column")
            raise ValueError("Invalid dataset format")

        self.load_model()

        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/training",
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.experiments.vignette_training.batch_size,
            per_device_eval_batch_size=self.config.experiments.vignette_training.eval_batch_size,
            gradient_accumulation_steps=self.config.experiments.vignette_training.gradient_accumulation_steps,
            warmup_steps=self.config.experiments.vignette_training.warmup_steps,
            weight_decay=self.config.experiments.vignette_training.weight_decay,
            learning_rate=self.config.experiments.vignette_training.learning_rate,
            logging_dir=f"{self.config.experiments.paths.logs_dir}/training",
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=self.config.experiments.vignette_training.eval_steps,
            save_strategy="steps",
            save_steps=self.config.experiments.vignette_training.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            save_total_limit=3,
            report_to=[],
            label_smoothing_factor=self.config.experiments.vignette_training.label_smoothing_factor,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            gradient_checkpointing=True,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            return_tensors="pt",
            padding=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.experiments.vignette_training.early_stopping_patience)]
        )

        print("Starting training...")
        try:
            trainer.train()
            final_model_path = f"{self.output_dir}/final_model"
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            self._save_training_metrics(trainer, "training")
            print(f"✅ Training completed! Model saved to: {final_model_path}")
            return trainer
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

    def _save_training_metrics(self, trainer, phase_name):
        """Save training metrics"""
        try:
            metrics = {
                'phase': phase_name,
                'final_train_loss': trainer.state.log_history[-1].get('train_loss', 0),
                'final_eval_loss': trainer.state.log_history[-1].get('eval_loss', 0),
                'total_steps': trainer.state.global_step,
                'epochs_completed': trainer.state.epoch,
                'best_model_checkpoint': trainer.state.best_model_checkpoint,
                'log_history': trainer.state.log_history[-10:],
                'timestamp': trainer.state.log_history[-1].get('epoch', 0)
            }
            metrics_file = Path(self.output_dir) / f'{phase_name}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"✅ Metrics saved to: {metrics_file}")
        except Exception as e:
            print(f"⚠️ Failed to save metrics: {e}")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("🔧 HYDRA CONFIGURATION DEBUG")
    print("=" * 50)
    print("✅ MAIN FUNCTION STARTED - LOGGING SHOULD WORK NOW")
    print("Current working directory:", os.getcwd())
    print("Config path resolved to:", hydra.core.config_store.ConfigStore.instance().repo)
    print("=" * 50)
    
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print("PRIORITY FIXES: ENHANCED TRAINING PIPELINE")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    try:
        # Disable struct mode to allow dynamic key access
        OmegaConf.set_struct(cfg, False)
        
        trainer = MedicalT5Trainer(cfg)
        os.makedirs(trainer.output_dir, exist_ok=True)
        print("\n🔧 PRIORITY FIXES APPLIED:")
        print("=" * 40)
        print("✅ Removed multitask learning")
        print("✅ Removed PubMed training")
        print("✅ Simplified training")
        print("✅ Consistent tokenizer handling")
        print("✅ Added validation")
        print("✅ Enhanced config validation")
        print("✅ Disabled struct mode for dynamic config access")
        print("✅ Fixed config path access with experiments prefix")
        print("=" * 40)
        trainer.train()
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model saved to: {trainer.output_dir}/final_model")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
