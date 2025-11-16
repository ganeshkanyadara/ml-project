import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# CONFIGURATION
MODEL_NAME = "meta-llama/Llama-3.2-3B"
DATASET_PATH = "samanantar_hindi_140.json"  # Use full dataset for better results

# Intersection neuron files
INTERSECTION_EN_PATH = "intersection_neurons_en.json"  # 23 neurons
INTERSECTION_HI_PATH = "intersection_neurons_hi.json"  # 170 neurons

OUTPUT_DIR_EN = "./finetuned_lora_english_intersection"
OUTPUT_DIR_HI = "./finetuned_lora_hindi_intersection"

# Memory-efficient settings
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10  # More epochs since we have fewer neurons
MAX_LENGTH = 256
GRADIENT_ACCUMULATION_STEPS = 4

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

print("="*80)
print("INTERSECTION NEURON FINE-TUNING")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Method: LoRA on Translation ∩ Language-Specific Neurons")
print(f"English intersection: {INTERSECTION_EN_PATH}")
print(f"Hindi intersection: {INTERSECTION_HI_PATH}")
print("="*80)


def load_intersection_neurons(path, language_name):
    
    print(f"\nLoading {language_name} intersection neurons from {path}...")
    
    try:
        with open(path, 'r') as f:
            intersection_mask = json.load(f)
        
        # Count total neurons
        total_neurons = sum(len(layer) for layer in intersection_mask)
        
        # Identify layers with neurons
        layers_with_neurons = [i for i, neurons in enumerate(intersection_mask) if len(neurons) > 0]
        
        print(f" {language_name} intersection loaded:")
        print(f"  - Total neurons: {total_neurons}")
        print(f"  - Layers with neurons: {len(layers_with_neurons)}")
        print(f"  - Layer indices: {layers_with_neurons}")
        
        return intersection_mask, total_neurons
    
    except FileNotFoundError:
        print(f" ERROR: File not found: {path}")
        print(f"\nYou need to generate intersection neurons first!")
        print(f"Make sure you have:")
        print(f"  - {INTERSECTION_EN_PATH}")
        print(f"  - {INTERSECTION_HI_PATH}")
        exit(1)


def get_target_layers_and_neurons(intersection_mask):
    """
    Get layers with intersection neurons and their neuron indices
    Returns: list of (layer_index, neuron_indices) tuples
    """
    target_info = []
    for layer_idx, neurons in enumerate(intersection_mask):
        if len(neurons) > 0:
            target_info.append((layer_idx, neurons))
    
    return target_info


def train_intersection_lora(language, output_dir, intersection_mask, total_neurons):
    """Train LoRA on intersection neuron layers"""
    
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*80}")
    print(f"TRAINING {language.upper()} INTERSECTION MODEL")
    print(f"{'='*80}")
    
    # Get target layers
    print(f"\n1. Analyzing intersection neurons...")
    target_info = get_target_layers_and_neurons(intersection_mask)
    
    if not target_info:
        print(f" No intersection neurons found for {language}!")
        return False
    
    target_layers = [layer_idx for layer_idx, _ in target_info]
    print(f"  Target layers: {target_layers}")
    print(f"  Total neurons: {total_neurons}")
    
    # Show neuron distribution
    print(f"\n  Neuron distribution per layer:")
    for layer_idx, neurons in target_info:
        print(f"    Layer {layer_idx:2d}: {len(neurons):3d} neurons - indices: {neurons[:5]}{'...' if len(neurons) > 5 else ''}")
    
    # Load dataset
    print(f"\n2. Loading dataset...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_texts = []
    for item in data:
        src = item['input']['src']
        tgt = item['input']['tgt']
        text = f"Translate English to Hindi:\n{src}\n\nHindi:\n{tgt}"
        train_texts.append(text)
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    print(f"  Dataset size: {len(train_dataset)} examples")
    
    # Load tokenizer
    print(f"\n3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Tokenizer loaded")
    
    # Load base model with 8-bit quantization
    print(f"\n4. Loading model with 8-bit quantization...")
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Base model loaded")
    
    # Create target modules for LoRA
    print(f"\n5. Configuring LoRA for {language} intersection neurons...")
    
    # CRITICAL: Only train MLP layers that contain intersection neurons
    target_modules = []
    for layer_idx in target_layers:
        # Target the MLP components where intersection neurons exist
        target_modules.append(f"model.layers.{layer_idx}.mlp.gate_proj")
        target_modules.append(f"model.layers.{layer_idx}.mlp.up_proj")
        target_modules.append(f"model.layers.{layer_idx}.mlp.down_proj")
    
    print(f"  Intersection-specific modules: {len(target_modules)} modules")
    print(f"  Example modules: {target_modules[:3]}")
    
    # LoRA config targeting ONLY intersection neuron layers
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"   LoRA applied")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.4f}%)")
    
    # Tokenize dataset
    print(f"\n6. Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
    
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    print(f"   Tokenization complete")
    
    # Training arguments
    print(f"\n7. Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        fp16=True,
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print(f"   Trainer ready")
    print(f"  Total training steps: {len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
    
    # Train
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING - {language.upper()} INTERSECTION")
    print(f"{'='*80}\n")
    
    try:
        trainer.train()
        print(f"\n{'='*80}")
        print(f" TRAINING COMPLETED - {language.upper()}")
        print(f"{'='*80}")
    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save model
    print(f"\n8. Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save detailed metadata
    info = {
        "language": language,
        "method": "LoRA (Translation ∩ Language-Specific Neurons)",
        "intersection_type": f"Translation ∩ {language}-specific",
        "total_neurons": total_neurons,
        "target_layers": target_layers,
        "layer_neuron_counts": {
            str(layer_idx): len(neurons) 
            for layer_idx, neurons in target_info
        },
        "target_modules": target_modules,
        "lora_config": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
        },
        "training_config": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dataset_size": len(train_dataset)
        }
    }
    
    with open(os.path.join(output_dir, "neuron_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"   Model saved to {output_dir}")
    
    # Cleanup
    del model, tokenizer, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return True


# MAIN
if __name__ == "__main__":
    
    # Check CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load intersection neurons
    en_mask, en_total = load_intersection_neurons(INTERSECTION_EN_PATH, "English")
    hi_mask, hi_total = load_intersection_neurons(INTERSECTION_HI_PATH, "Hindi")
    
    print(f"\n{'='*80}")
    print("INTERSECTION SUMMARY")
    print(f"{'='*80}")
    print(f"English intersection: {en_total} neurons (Translation ∩ English-specific)")
    print(f"Hindi intersection: {hi_total} neurons (Translation ∩ Hindi-specific)")
    print(f"Ratio: 1:{hi_total/en_total:.1f} (Hindi has {hi_total/en_total:.1f}x more neurons)")
    
    # Train English intersection model
    print("\n" + "="*80)
    print("PHASE 1: ENGLISH INTERSECTION MODEL")
    print("="*80)
    success_en = train_intersection_lora("English", OUTPUT_DIR_EN, en_mask, en_total)
    
    # Train Hindi intersection model
    print("\n" + "="*80)
    print("PHASE 2: HINDI INTERSECTION MODEL")
    print("="*80)
    success_hi = train_intersection_lora("Hindi", OUTPUT_DIR_HI, hi_mask, hi_total)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"English Intersection ({en_total} neurons): {' SUCCESS' if success_en else ' FAILED'}")
    print(f"Hindi Intersection ({hi_total} neurons):   {' SUCCESS' if success_hi else ' FAILED'}")
    
    if success_en and success_hi:
        
    
        print(f"\nModels saved:")
        print(f"  - {OUTPUT_DIR_EN}/")
        print(f"  - {OUTPUT_DIR_HI}/")
        
    
    print("\n" + "="*80)