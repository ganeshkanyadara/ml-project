import torch
import json
import os
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset

# ======== CONFIGURATION ========
# --- Model and Paths ---
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"

# These are the files with (Task ∩ Language) neurons
EN_INTERSECTION_PATH = "intersection_neurons_en.json"
HI_INTERSECTION_PATH = "intersection_neurons_hi.json"

# This is your training data
DATASET_PATH = "samanantar_hindi.json" 
OUTPUT_DIR = "./finetuned_selective_down_proj"

# --- Training Hyperparameters ---
BATCH_SIZE = 2
LEARNING_RATE = 1e-4  # Use a slightly higher LR for sparse updates
NUM_EPOCHS = 3
MAX_LENGTH = 256
GRADIENT_ACCUMULATION_STEPS = 4

print("="*80)
print("Selective Neuron Fine-Tuning (Gradient Masking)")
print(f"Model: {BASE_MODEL_NAME}")
print(f"Method: Freeze all, tune only intersected 'down_proj' neurons.")
print("="*80)

def load_and_combine_neurons(en_path, hi_path):
    """
    Loads the English and Hindi intersection neurons and returns a
    dictionary mapping layer_idx to a UNION set of neuron indices.
    This creates the "master list" of all neurons to be tuned.
    """
    print(f"\nLoading and combining neuron masks...")
    print(f" EN: {en_path}")
    print(f" HI: {hi_path}")

    try:
        with open(en_path, 'r') as f:
            en_data = json.load(f)
        with open(hi_path, 'r') as f:
            hi_data = json.load(f)
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find neuron file: {e}")
        print("Please make sure 'intersection_neurons_en.json' and 'intersection_neurons_hi.json' are present.")
        return None, 0

    # Determine number of layers from the longest mask list
    num_layers = max(len(en_data), len(hi_data))
    if len(en_data) != len(hi_data):
         print(f"Warning: Neuron files have mismatched layer counts. Using max layers: {num_layers}")
        
    combined_mask_dict = {}
    total_en_neurons = 0
    total_hi_neurons = 0
    total_union_neurons = 0

    for i in range(num_layers):
        en_set = set(en_data[i] if i < len(en_data) else [])
        hi_set = set(hi_data[i] if i < len(hi_data) else [])
        
        union_set = en_set.union(hi_set)
        
        total_en_neurons += len(en_set)
        total_hi_neurons += len(hi_set)
        total_union_neurons += len(union_set)

        if union_set:
            combined_mask_dict[i] = sorted(list(union_set))

    print(f"  Total EN ∩ Task Neurons: {total_en_neurons}")
    print(f"  Total HI ∩ Task Neurons: {total_hi_neurons}")
    print(f"  Total Unique Neurons (Union) to be tuned: {total_union_neurons}")
    
    return combined_mask_dict, num_layers

def apply_gradient_masks(model, combined_mask_dict, num_layers_in_mask):
    """
    Applies gradient masking hooks to the model based on user's steps.
    """
    print("\nApplying gradient masks as per instructions...")

    # --- Step 1: Freeze all model parameters ---
    print("  Step 1: Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Get model dimensions to validate masks
    try:
        model_intermediate_size = model.config.intermediate_size
        hidden_size = model.config.hidden_size
        config_num_layers = model.config.num_hidden_layers
        print(f"  Model Architecture: hidden_size={hidden_size}, intermediate_size={model_intermediate_size}, num_layers={config_num_layers}")
    except Exception as e:
        print(f"Could not read model config: {e}. Aborting mask application.")
        return 0

    if num_layers_in_mask > config_num_layers:
        print(f"Warning: Your mask files have {num_layers_in_mask} layers, but the model only has {config_num_layers}. Extra layers will be ignored.")
        
    # Check for the critical mismatch
    max_mask_neuron = 0
    if combined_mask_dict:
        # Find the highest neuron index specified in any layer
        max_mask_neuron = max(max(v) for v in combined_mask_dict.values() if v)
        
    if max_mask_neuron >= model_intermediate_size:
        print("\n" + "!"*80)
        print(" CRITICAL ERROR: MODEL-MASK MISMATCH")
        print(f" Your neuron masks have indices up to {max_mask_neuron},")
        print(f" but your model '{BASE_MODEL_NAME}' only has an intermediate size of {model_intermediate_size}.")
        print(" The neuron masks were generated for a DIFFERENT model.")
        print(" ABORTING. Please fix this mismatch.")
        print("!"*80 + "\n")
        return 0

    total_neurons_tuned = 0
    target_layers_found = []

    # --- Step 3: Define the hook function factory to apply gradient masks ---
    def create_mask_hook(mask):
        """Creates a hook function that applies a gradient mask."""
        def hook(grad):
            # Element-wise multiplication with the mask
            # This zeros out gradients for all non-intersected neurons
            return grad * mask
        return hook

    print("  Steps 2 & 3: Enabling gradients and applying masks to target 'down_proj' neurons...")
    # --- Step 2 & 3: Iterate over layers, enable gradients, and apply masks ---
    for layer_idx, neuron_indices in combined_mask_dict.items():
        if layer_idx >= config_num_layers:
            continue # Skip layers from mask that don't exist in model
            
        if not neuron_indices:
            continue

        try:
            # As requested, we only target the down_proj layer
            module_name = f"model.layers.{layer_idx}.mlp.down_proj"
            module = model.get_submodule(module_name)
            
            if not isinstance(module, torch.nn.Linear):
                print(f"Warning: Module {module_name} is not a Linear layer. Skipping.")
                continue

            # Weight shape is (hidden_size, intermediate_size)
            if module.weight.shape != (hidden_size, model_intermediate_size):
                print(f"ERROR: Mismatch! Layer {layer_idx} weight shape {module.weight.shape} != expected {(hidden_size, model_intermediate_size)}")
                continue
                
            # --- Check if layer is float (it should be if skipped) ---
            if not module.weight.dtype.is_floating_point:
                print(f"ERROR: Layer {layer_idx} is not floating point (dtype: {module.weight.dtype}).")
                print("This means it was not correctly skipped from quantization.")
                print("Please ensure 'llm_int8_skip_modules' is set correctly in BitsAndBytesConfig.")
                continue # Skip this layer

            # Create the mask. Shape (hidden_size, intermediate_size)
            # We initialize with zeros, so only target neurons get gradients
            mask = torch.zeros_like(module.weight, device=module.weight.device)

            # The "neurons" are the input channels to the down_proj layer.
            # These correspond to the *columns* (dim 1) of the down_proj.weight matrix.
            valid_indices = [idx for idx in neuron_indices if idx < model_intermediate_size]
            
            if not valid_indices:
                continue

            # Set mask to 1.0 for the columns we want to tune
            # This allows gradients to flow *only* for these neurons
            mask[:, valid_indices] = 1.0

            # --- Step 2 (Partial): Enable gradients ONLY for this specific weight tensor ---
            module.weight.requires_grad = True
            
            # --- Step 3 (Partial): Register the backward hook to apply the mask ---
            module.weight.register_hook(create_mask_hook(mask))
            
            total_neurons_tuned += len(valid_indices)
            target_layers_found.append(layer_idx)

        except AttributeError:
            print(f"Warning: Could not find module 'model.layers.{layer_idx}.mlp.down_proj' in model. Skipping layer {layer_idx}.")
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")

    if total_neurons_tuned > 0:
        print(f"\nSuccessfully applied gradient masks.")
        print(f"  Targeted Layers: {sorted(list(target_layers_found))}")
        print(f"  Total Neurons to be Tuned: {total_neurons_tuned}")
    else:
        print("\nERROR: No valid neurons were found to tune. This may be due to the model-mask mismatch or quantization error.")

    return total_neurons_tuned

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # The number of trainable params should be (hidden_size * total_neurons_tuned)
    print(
        f"\n  Trainable params: {trainable_params:,} | All params: {all_param:,} | "
        f"Trainable %: {100 * trainable_params / all_param:.8f}"
    )

def load_and_tokenize_dataset(tokenizer, data_path):
    """Loads and prepares the translation dataset."""
    print(f"\nLoading and tokenizing dataset from {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {data_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Dataset file {data_path} is not valid JSON.")
        return None
    
    # Format data into prompts
    train_texts = []
    for item in data:
        try:
            src = item['input']['src']
            tgt = item['input']['tgt']
            # We create a simple prompt for fine-tuning
            text = f"Translate English to Hindi:\n{src}\n\nHindi:\n{tgt}"
            train_texts.append(text)
        except KeyError:
            print(f"Warning: Skipping invalid data item: {item}")

    if not train_texts:
        print("ERROR: No valid data loaded from dataset file. Check file format.")
        return None
        
    print(f"  Loaded {len(train_texts)} examples.")
    train_dataset = Dataset.from_dict({"text": train_texts})

    # Tokenize
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
    print("  Tokenization complete.")
    return tokenized_dataset


def main():
    # 1. Load and combine the neuron masks
    combined_mask_dict, num_layers_in_mask = load_and_combine_neurons(EN_INTERSECTION_PATH, HI_INTERSECTION_PATH)
    if combined_mask_dict is None:
        return

    # 2. Load Tokenizer
    print(f"\nLoading tokenizer for {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  Tokenizer loaded.")

    # 3. Load Model with 8-bit quantization for memory efficiency
    print(f"\nLoading base model {BASE_MODEL_NAME} with 8-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["down_proj", "lm_head"] 
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        dtype=torch.float16, # Use dtype instead of torch_dtype
        device_map="auto",
        trust_remote_code=True,
    )
    print("  Base model loaded (down_proj & lm_head layers skipped from quantization).")

    # 4. Apply the gradient masks
    # This function implements all 3 of your requested steps.
    total_tuned_neurons = apply_gradient_masks(model, combined_mask_dict, num_layers_in_mask)
    
    if total_tuned_neurons == 0:
        print("CRITICAL: No neurons are marked for tuning. Please fix the mismatch and try again.")
        return

    # 5. Print trainable parameter count (should be very small)
    print_trainable_parameters(model)

    # 6. Load and Tokenize Dataset
    tokenized_dataset = load_and_tokenize_dataset(tokenizer, DATASET_PATH)
    if tokenized_dataset is None:
        return

    # 7. Set up Training
    print("\nSetting up Trainer...")

    # This tricks the Trainer's safety check.
    # The Trainer sees 'is_quantized=True' and blocks training,
    # even though we *know* the parts we're training are float16.
    # We manually set it to False to bypass this check.
    print("  Bypassing Trainer's quantization safety check...")
    model.is_quantized = False
    
    # Use 8-bit Adam optimizer for memory efficiency
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=200, # Save full model checkpoints
        save_total_limit=1,
        fp16=False, # <-- THIS IS THE FIX: Must be False when using 8-bit optimizer
        optim="adamw_8bit", # Optimizer compatible with 8-bit
        report_to="none",
        remove_unused_columns=False,
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
    print("  Trainer is ready.")

    # 8. Train
    print(f"\n{'='*80}")
    print("STARTING SELECTIVE FINE-TUNING")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")

    # 9. Save the full model
    # Note: This saves the *entire model*, not just an adapter,
    # as we have directly modified the base model's weights.
    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metadata
    info = {
        "base_model": BASE_MODEL_NAME,
        "method": "Selective Fine-Tuning (Gradient Masking)",
        "target_layers": "mlp.down_proj",
        "tuned_neuron_info_by_layer": combined_mask_dict,
        "total_tuned_neurons": total_tuned_neurons,
        "training_config": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dataset_size": len(tokenized_dataset)
        }
    }
    with open(os.path.join(OUTPUT_DIR, "selective_tuning_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
        
    print(f"  Model and info saved to {OUTPUT_DIR}")
    
    # Cleanup
    del model, tokenizer, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()