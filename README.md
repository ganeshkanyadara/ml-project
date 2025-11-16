# **LLaMA Neuron Interpretation Project**

This repository contains two complementary research pipelines that explore how neurons inside LLaMA models behave under different conditions. The goal is to understand which neurons encode language identity, which support downstream tasks, and how masking these neurons affects model behavior.

The project is divided into two major components:

1. **Language-Specific Neurons** – implemented in *Google Colab*, with all outputs saved to *Google Drive*.
2. **Task-Specific Neurons** – implemented *locally* on your machine using Jupyter/VS Code.

Both pipelines follow a similar structure (identify → score → mask → evaluate) but differ in goals and execution environments.

---

# **1. Language-Specific Neurons**

### *Goal:* Identify neurons that fire strongly for specific languages (English, Hindi) and test how masking them changes generation.

### **Notebooks:**

| Notebook                              | Purpose                                                                                 | Execution        |
| ------------------------------------- | --------------------------------------------------------------------------------------- | ---------------- |
| `Language_specific_Neurons.ipynb` | Full pipeline for extracting language-dependent neuron activations and generating masks | **Google Colab** |



### **Outputs Stored in Google Drive**

```
/content/drive/MyDrive/Colab_Projects/Language-Specific-Neurons/
    activation_masks/
    dataset/
    results/
    data/
```

### **Pipeline Summary**

1. Load LLaMA model in Colab
2. Collect neuron activations for English & Hindi
3. Identify language specific neurons
4. Compute language-difference statistics through ppl scores.
5. Generate neuron masks for each language
6. Apply masks during inference to observe behavior change

This reveals whether certain neurons encode *language-specific structure* inside the model.

---

# **2. Task-Specific Neurons (Local Machine)**

### *Goal:* Identify neurons that activate for specific downstream tasks (classification, reasoning, sentiment, etc.) and evaluate their causal impact.

### **Notebooks:**

| Notebook             | Purpose                                               | Execution |
| -------------------- | ----------------------------------------------------- | --------- |
| `identify.ipynb`     | Extract task-based activations and rank neurons       | **Local** |
| `evaluation.ipynb`   | Test model performance after masking task neurons     | **Local** |
| `orginal_code.ipynb` | Original baseline implementation of the full pipeline | **Local** |


### **Outputs Saved Locally**

```
task_neuron_project/
    new_folder/
    identify.ipynb
    activation.pth
    evaluation.ipynb
    original_cpde.ipynb
```

### **Pipeline Summary**

1. Load task dataset
2. Capture neuron activations while model performs the task
3. Compute neuron importance scores by taking 5% neurons
4. Create masks of task-relevant neurons
5. Evaluate performance with/without those neurons

This reveals whether certain neurons encode *task-specific information* inside the model.


---

# **3. How the Two Projects Connect**

Although they target different aspects of model behavior, both pipelines share a common interpretability philosophy:

| Language-specific neurons                          | Task-specific neurons                                   |
| -------------------------------------------------- | --------------------------------------------------------|
| Reveal which neurons represent linguistic identity | Reveal which neurons support language translation tasks |
| Focus on multilingual internal structure           | Focus on model capability and reasoning                 |
| Uses language difference metrics                   | Uses task activation deltas                             |
| Masking causes language drift                      | Masking causes performance degradation                  |

Together, they provide a larger picture of:

### ✔ Where linguistic information is stored

### ✔ Where task abilities emerge

### ✔ How neuron-level interventions impact LLaMA behavior

---

# **4. Requirements**

### **Language-Specific Neurons (Colab)**

Handled automatically inside Colab:

* PyTorch (CUDA-enabled)
* Transformers
* Accelerate
* tqdm
* Google Drive mounted

### **Task-Specific Neurons (Local)**

Create a venv:

```bash
python3 -m venv neuronenv
source neuronenv/bin/activate
```

Install packages:

```bash
pip install torch transformers accelerate numpy tqdm matplotlib
```

---

# **5. License**

Open for academic research, model interpretability work, and experimentation.

