import json
import torch

import os

# Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}\n")

# Load the language neurons (PyTorch tensors)
# Update these paths to match your actual file locations
try:
    lang_neurons = torch.load("llama32_activation_masks (1).pt", weights_only=False)
    print(" Loaded language neurons from 'llama32_activation_masks (1).pt'")
except FileNotFoundError:
    print(" File 'llama32_activation_masks (1).pt' not found")
    print("\nPlease update the file path in the script to match your file location.")
    print("Looking for .pt or .pth files in current directory:")
    for file in os.listdir('.'):
        if file.endswith(('.pt', '.pth')):
            print(f"  - {file}")
    exit(1)

lang_neurons_en = lang_neurons[0]  # English neurons
lang_neurons_hi = lang_neurons[1]  # Hindi neurons

# Load the task neurons
try:
    task_neurons = torch.load("activation_mask.pth", weights_only=False)
    print(" Loaded task neurons from 'activation_mask.pth'")
except FileNotFoundError:
    print(" File 'activation_mask.pth' not found")
    print("\nPlease update the file path in the script.")
    exit(1)

task_neurons = task_neurons[0]
print()

def find_intersection_neurons(lang_neurons, task_neurons):
    
    intersections = []
    
    num_layers = len(lang_neurons)
    
    for i in range(num_layers):
        # Convert tensors to sets
        lang_set = set(lang_neurons[i].tolist())
        task_set = set(task_neurons[i].tolist())
        
        # Find intersection
        inter = lang_set.intersection(task_set)
        
        # Convert to sorted list
        intersections.append(sorted(list(inter)))
    
    return intersections

# Find intersections for English
print("=" * 60)
print("FINDING INTERSECTION NEURONS FOR ENGLISH")
print("=" * 60)
intersections_en = find_intersection_neurons(lang_neurons_en, task_neurons)

# Find intersections for Hindi
print("\n" + "=" * 60)
print("FINDING INTERSECTION NEURONS FOR HINDI")
print("=" * 60)
intersections_hi = find_intersection_neurons(lang_neurons_hi, task_neurons)

# Save to JSON files
with open('intersection_neurons_en.json', 'w') as f:
    json.dump(intersections_en, f, indent=2)

with open('intersection_neurons_hi.json', 'w') as f:
    json.dump(intersections_hi, f, indent=2)

# Print statistics for English
print("\n" + "=" * 60)
print("ENGLISH INTERSECTION NEURONS STATISTICS")
print("=" * 60)
total_en = 0
for i, inter in enumerate(intersections_en):
    total_en += len(inter)
    print(f"Layer {i:2d}: {len(inter):4d} neurons - {inter[:5]}{'...' if len(inter) > 5 else ''}")

print("=" * 60)
print(f"Total English intersection neurons: {total_en}")

# Print statistics for Hindi
print("\n" + "=" * 60)
print("HINDI INTERSECTION NEURONS STATISTICS")
print("=" * 60)
total_hi = 0
for i, inter in enumerate(intersections_hi):
    total_hi += len(inter)
    print(f"Layer {i:2d}: {len(inter):4d} neurons - {inter[:5]}{'...' if len(inter) > 5 else ''}")

print("=" * 60)
print(f"Total Hindi intersection neurons: {total_hi}")

# Compare English and Hindi overlap
print("\n" + "=" * 60)
print("COMPARISON: ENGLISH vs HINDI OVERLAP")
print("=" * 60)
for i in range(len(intersections_en)):
    en_count = len(intersections_en[i])
    hi_count = len(intersections_hi[i])
    diff = en_count - hi_count
    print(f"Layer {i:2d}: EN={en_count:4d}, HI={hi_count:4d}, Diff={diff:+4d}")
