import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    import sacrebleu
except:
    print("Installing sacrebleu...")
    import subprocess
    subprocess.run(["pip", "install", "sacrebleu"], check=True)
    import sacrebleu

# ==== CONFIG ====
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"

# UPDATED: Use intersection model paths
LORA_MODEL_PATH_EN = "./finetuned_lora_english_intersection"
LORA_MODEL_PATH_HI = "./finetuned_lora_hindi_intersection"

DATASET_PATH = "samanantar_hindi_10.json"

GEN_MAX_NEW_TOKENS = 64
GEN_NUM_BEAMS = 4
GEN_DO_SAMPLE = False

# Fallback test sentences
TEST_SENTENCES = [
    ("The cat is sleeping.", "बिल्ली सो रही है।"),
    ("I love books.", "मुझे किताबें पसंद हैं।"),
    ("The court has fixed a hearing for February 12", "अदालत ने इस मामले में आगे की सुनवाई के लिए एक फरवरी की तारीख़ तय की"),
    ("Machine learning is a subset of artificial intelligence.", None),
    ("The weather is beautiful today.", None),
    ("Please send me the report by email.", None),
]

print("="*80)
print("INTERSECTION NEURON MODEL COMPARISON")
print("="*80)
print(f"Base Model: {BASE_MODEL_NAME}")
print(f"English Intersection Model: {LORA_MODEL_PATH_EN}")
print(f"Hindi Intersection Model: {LORA_MODEL_PATH_HI}")
print("="*80)


def normalize_text(s):
    
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([।?!,.:;])", r"\1", s)
    return s


def corpus_bleu_score(references, hypotheses):
    
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return float(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score)


def load_dataset():
    
    print(f"\nLoading dataset from {DATASET_PATH}...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset file not found. Using fallback TEST_SENTENCES.")
        return TEST_SENTENCES

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback TEST_SENTENCES.")
        return TEST_SENTENCES

    # Handle list of dicts format (your samanantar_hindi_10.json format)
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        
        # Format: [{"id": ..., "input": {"src": "...", "tgt": "..."}}]
        if isinstance(first, dict) and 'input' in first:
            pairs = []
            for item in data:
                src = item['input']['src']
                tgt = item['input']['tgt']
                pairs.append((src, tgt))
            print(f" Loaded {len(pairs)} sentence pairs")
            return pairs
        
        # Format: [["en", "hi"], ...]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            print(f" Loaded {len(data)} sentence pairs")
            return data
    
    # Fallback
    print("Could not interpret dataset format. Using fallback TEST_SENTENCES.")
    return TEST_SENTENCES


def load_base_model_and_tokenizer():
    
    print("\nLoading base model...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()
    print(" Base model loaded")
    return base, tokenizer


def load_finetuned_peft(path, name):
    
    if not os.path.exists(path):
        print(f" Model not found at: {path}")
        return None, None
    
    print(f"\nLoading {name} model from {path}...")
    
    # Load base model
    base, tokenizer = load_base_model_and_tokenizer()
    
    # Load LoRA adapter
    try:
        model = PeftModel.from_pretrained(base, path)
        model.eval()
    except Exception as e:
        print(f" Failed to load LoRA adapter: {e}")
        return None, None
    
    # Load and display neuron info
    info_path = os.path.join(path, "neuron_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"  Model Info:")
            print(f"    - Total neurons: {info['total_neurons']}")
            print(f"    - Intersection type: {info.get('intersection_type', 'N/A')}")
            print(f"    - Target layers: {info['target_layers']}")
        except Exception as e:
            print(f"  (Could not load neuron info: {e})")
    
    print(f" {name} model loaded")
    return model, tokenizer


def build_prompt(text):
    
    return f"Translate the following English sentence to fluent Hindi. Output only the Hindi translation.\n\nEnglish: {text}\nHindi:"


def translate(model, tokenizer, text, max_new_tokens=GEN_MAX_NEW_TOKENS):
    
    prompt = build_prompt(text)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=GEN_DO_SAMPLE,
            num_beams=GEN_NUM_BEAMS,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = out[0]
    input_len = enc["input_ids"].shape[1]
    
    # Handle different tensor shapes
    if generated.ndim == 1:
        gen_ids = generated[input_len:]
    else:
        gen_ids = generated[0][input_len:]
    
    # Decode
    if gen_ids.numel() == 0:
        # No new tokens generated, extract from full output
        full = tokenizer.decode(generated, skip_special_tokens=True)
        if "Hindi:" in full:
            translation = full.split("Hindi:")[-1].strip()
        else:
            translation = full[len(prompt):].strip()
    else:
        translation = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Return first non-empty line
    for line in translation.splitlines():
        if line.strip():
            return line.strip()
    
    return translation


def main():
    # Load dataset
    pairs = load_dataset()
    print(f"\n{'='*80}")
    print(f"Loaded {len(pairs)} sentence pair(s) for evaluation")
    print(f"{'='*80}")
    
    # Load base model
    base_model, base_tokenizer = load_base_model_and_tokenizer()
    
    # Load fine-tuned models
    ft_models = []
    
    # Try loading English intersection model
    en_model, en_tokenizer = load_finetuned_peft(
        LORA_MODEL_PATH_EN, 
        "English-Intersection (23 neurons)"
    )
    if en_model is not None:
        ft_models.append(("English-Neurons", en_model, en_tokenizer or base_tokenizer))
    
    # Try loading Hindi intersection model
    hi_model, hi_tokenizer = load_finetuned_peft(
        LORA_MODEL_PATH_HI,
        "Hindi-Intersection (170 neurons)"
    )
    if hi_model is not None:
        ft_models.append(("Hindi-Neurons", hi_model, hi_tokenizer or base_tokenizer))
    
    if not ft_models:
        print("\n" + "="*80)
        print("ERROR: No fine-tuned models found!")
        print("="*80)
        print("\nPlease train the models first:")
        print("  python finetune_intersection_neurons.py")
        print("\nExpected model locations:")
        print(f"  - {LORA_MODEL_PATH_EN}")
        print(f"  - {LORA_MODEL_PATH_HI}")
        return
    
    # Initialize results storage
    results = {
        "baseline": [],
        "finetuned": {name: [] for name, _, _ in ft_models}
    }
    
    # Test all models
    print("\n" + "="*80)
    print("TRANSLATION COMPARISON")
    print("="*80)
    
    for i, (src, ref) in enumerate(pairs):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}/{len(pairs)}")
        print(f"{'='*80}")
        print(f"EN:  {src}")
        if ref is not None:
            print(f"REF: {ref}")
        
        # Baseline translation
        base_out = translate(base_model, base_tokenizer, src)
        print(f"BASE: {base_out}")
        results["baseline"].append(base_out)
        
        # Fine-tuned model translations
        for name, model, tokenizer in ft_models:
            out = translate(model, tokenizer, src)
            print(f"{name}: {out}")
            results["finetuned"][name].append(out)
    
    # Calculate BLEU scores if references exist
    ref_indices = [i for i, (_, r) in enumerate(pairs) if r is not None]
    
    metrics = {}
    
    if ref_indices:
        refs = [pairs[i][1] for i in ref_indices]
        baseline_hyps = [results["baseline"][i] for i in ref_indices]
        
        print(f"\n{'='*80}")
        print("BLEU SCORES (sacreBLEU)")
        print(f"{'='*80}")
        
        baseline_bleu = corpus_bleu_score(refs, baseline_hyps)
        print(f"\nBaseline sacreBLEU: {baseline_bleu:.2f}")
        
        metrics["baseline_sacrebleu"] = baseline_bleu
        
        finetuned_bleus = {}
        for name in results["finetuned"]:
            hyps = [results["finetuned"][name][i] for i in ref_indices]
            bleu = corpus_bleu_score(refs, hyps)
            finetuned_bleus[name] = bleu
            
            print(f"{name} sacreBLEU: {bleu:.2f}  (Δ {bleu - baseline_bleu:+.2f})")
            
            # Check for identical outputs
            identical_to_base = sum(
                1 for a, b in zip(baseline_hyps, hyps) 
                if normalize_text(a) == normalize_text(b)
            )
            
            if identical_to_base == len(hyps):
                print(f"   WARNING: All {identical_to_base} outputs identical to baseline!")
                print(f"    This suggests the fine-tuning had no effect.")
            elif identical_to_base > 0:
                print(f"  ℹ Note: {identical_to_base}/{len(hyps)} outputs identical to baseline")
        
        metrics["finetuned_sacrebleu"] = finetuned_bleus
    else:
        print("\n No references found in dataset. BLEU scores cannot be computed.")
    
    # Save results
    output = {
        "test_sentences": [s for s, _ in pairs],
        "references": [r for _, r in pairs],
        "baseline_translations": results["baseline"],
        "finetuned_translations": results["finetuned"],
        "metrics": metrics,
        "model_info": {
            "base_model": BASE_MODEL_NAME,
            "english_intersection_neurons": 23,
            "hindi_intersection_neurons": 170,
            
        }
    }
    
    output_file = "comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SAVED")
    print(f"{'='*80}")
    print(f" {output_file}")
    
    # Print analysis
    if ref_indices:
        print(f"\n{'='*80}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nModel Comparison:")
        print(f"  Baseline:          {metrics['baseline_sacrebleu']:.2f} BLEU")
        for name, bleu in metrics['finetuned_sacrebleu'].items():
            delta = bleu - metrics['baseline_sacrebleu']
            trend = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  {name}: {bleu:.2f} BLEU ({trend} {abs(delta):.2f})")
        
        


if __name__ == "__main__":
    main()