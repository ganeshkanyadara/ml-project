import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import gc
import sacrebleu
from tqdm import tqdm
import random

#  CONFIGURATION 
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL_PATH = "./finetuned_selective_down_proj" 
TEST_DATA_PATH = "samanantar_hindi_10.json" 
NUM_TEST_SAMPLES = 8 

print("="*80)
print("FINAL MODEL EVALUATION")
print("="*80)

def get_bnb_config():
    """
    Critical: Must match training config to load 'down_proj' correctly.
    """
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["down_proj", "lm_head"] 
    )

def load_test_data(path, num_samples=20):
    """Loads a random subset of source and reference sentences."""
    print(f"Loading dataset: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {path} not found. Using dummy data.")
        return ["The cat sits."], ["बिल्ली बैठी है।"]

    # Filter for valid pairs
    valid_data = []
    for item in data:
        if 'input' in item and 'src' in item['input'] and 'tgt' in item['input']:
            valid_data.append( (item['input']['src'], item['input']['tgt']) )
    
    # Random sample
    if len(valid_data) > num_samples:
        sampled = random.sample(valid_data, num_samples)
    else:
        sampled = valid_data
        
    sources = [s for s, t in sampled]
    references = [t for s, t in sampled]
            
    print(f"Selected {len(sources)} random examples for evaluation.")
    return sources, references

def generate_translations(model, tokenizer, sources):
    """Generates translations using One-Shot Prompting."""
    hypotheses = []
    
    # One-Shot Prompt Template
    # This teaches the model the format: English -> Hindi
    prompt_template = (
        "Translate English to Hindi:\n"
        "English: The weather is very nice today.\n"
        "Hindi: आज मौसम बहुत सुहावना है।\n\n"
        "English: {src}\n"
        "Hindi:"
    )
    
    print("Generating translations...")
    for src in tqdm(sources):
        prompt = prompt_template.format(src=src)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False, # Deterministic (Greedy)
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse output
        try:
            # We want the text AFTER the last "Hindi:"
            if "Hindi:" in full_output:
                generated_text = full_output.rsplit("Hindi:", 1)[1].strip()
            else:
                generated_text = full_output
            
            # Clean up: take only the first line
            generated_text = generated_text.split('\n')[0].strip()
        except:
            generated_text = ""
            
        hypotheses.append(generated_text)
        
    return hypotheses

def evaluate_model(model_path, model_name, sources, references):
    """Loads model, translates, calculates score, and unloads."""
    print(f"\n{'='*40}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*40}")
    
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load Model
        print(f" Loading from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=get_bnb_config(),
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Generate
        hypotheses = generate_translations(model, tokenizer, sources)
        
        # Calculate BLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        
        print(f"\n {model_name} RESULTS:")
        print(f"   BLEU Score: {bleu.score:.2f}")
        
        # Print 3 examples
        print("\n   Examples:")
        for i in range(min(3, len(sources))):
            print(f"   Src: {sources[i]}")
            print(f"   Ref: {references[i]}")
            print(f"   Hyp: {hypotheses[i]}")
            print("-" * 30)

        # Cleanup to free GPU for next model
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return bleu.score

    except Exception as e:
        print(f" Error: {e}")
        return 0.0

# MAIN 
if __name__ == "__main__":
    # 1. Prepare Data
    sources, references = load_test_data(TEST_DATA_PATH, NUM_TEST_SAMPLES)
    
    # 2. Evaluate Baseline (Original Llama-3.2)
    baseline_score = evaluate_model(BASE_MODEL_NAME, "BASELINE", sources, references)
    
    # 3. Evaluate Fine-Tuned (Your Model)
    finetuned_score = evaluate_model(FINETUNED_MODEL_PATH, "FINE-TUNED", sources, references)

    # 4. Final Report
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Baseline BLEU:   {baseline_score:.2f}")
    print(f"Fine-Tuned BLEU: {finetuned_score:.2f}")
    
    diff = finetuned_score - baseline_score
    
    if diff > 0.5:
        print(f"\n SUCCESS: The model improved by +{diff:.2f} points!")
        print("   The selective fine-tuning worked.")
    elif diff < -0.5:
        print(f"\n REGRESSION: The model degraded by {diff:.2f} points.")
        print("   The training might have been unstable or overwritten useful knowledge.")
    else:
        print("\n NEUTRAL: No significant difference.")
        print("   The training signal might still be too weak (1e-5 might be too low for only 185 neurons).")