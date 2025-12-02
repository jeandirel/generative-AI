# Lab 2 Analysis & Implementation Plan

## Goal Description
The objective of Lab 2 is to gain practical experience with **LLM Fine-tuning**, specifically using **DistilGPT2** on the **Tiny Shakespeare** dataset. The lab explores two main approaches: **Full Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** (Low-Rank Adaptation). It also involves analyzing the impact of these methods on model performance (Perplexity), generation quality, and "catastrophic forgetting" of general knowledge.

## User Review Required
> [!IMPORTANT]
> **Compute Resources**: The "Ablation Study" (Part 8) and "Full Fine-tune" (Part 3) require training loops. While `distilgpt2` is small, training might take time depending on the available hardware (CPU vs GPU).
> **Deliverables**: The final output is the completed Jupyter Notebook with executed cells, filled-in results tables, and written analysis.

## Identified Tasks (What the Professor Expects)

The lab is divided into 8 parts. Here is the breakdown of expected work:

### 1. Data Preparation (Part 1)
- **Task**: Tokenize the "Tiny Shakespeare" dataset and pack it into fixed-length blocks (`block_size=256`).
- **Code**: Implement `tokenize` and `group_texts` functions.

### 2. Baseline Evaluation (Part 2)
- **Task**: Calculate the validation perplexity (PPL) of the pre-trained `distilgpt2` model *before* any fine-tuning.
- **Code**: Use the provided `compute_ppl` function.

### 3. Full Fine-Tuning (Part 3)
- **Task**: Fine-tune all parameters of `distilgpt2` for ~3-5 epochs.
- **Code**: Configure `TrainingArguments` and `Trainer`, then run training.
- **Output**: Validation PPL and a generation sample.

### 4. LoRA Fine-Tuning (Part 5)
- **Task**: Fine-tune using LoRA (updating only a small subset of parameters).
- **Code**: Configure `LoraConfig` (rank `r=16`), `TrainingArguments`, and `Trainer`.
- **Output**: Validation PPL.

### 5. Prompting Experiments (Part 6)
- **Task**: Compare text generation across 3 models: Baseline, Full FT, and LoRA.
- **Experiments**:
    - Zero-shot continuation (Shakespeare style).
    - Instruction-style (Modern English to Shakespeare?).
    - Few-shot pattern completion.

### 6. Catastrophic Forgetting Test (Part 7)
- **Task**: Test if the fine-tuned models have "forgotten" modern English/general knowledge.
- **Analysis**: Compare if LoRA retains more general knowledge than Full FT.

### 7. Ablation Study (Part 8)
- **Task**: Systematically test different LoRA ranks (`r=1`, `r=8`, `r=64`) to find the "sweet spot".
- **Output**: A table comparing Rank, Trainable Params %, and Validation PPL.

### 8. Final Report (Part 9)
- **Task**: Synthesize all results into a summary table and answer analysis questions.
- **Deliverable**: Completed markdown sections in the notebook.

## Proposed Changes (Implementation Steps)

I will implement the missing code in `Lab2_LLM_Finetune_with_code.ipynb`.

### [Notebook] Lab2_LLM_Finetune_with_code.ipynb
#### [MODIFY] Part 1: Tokenization & Packing
- Implement `tokenize(ex)` to process the "Text" column.
- Implement `group_texts(examples)` to concatenate and chunk tokens.

#### [MODIFY] Part 3: Full Fine-tune
- Adjust `num_train_epochs` to 5 (as suggested in TODO).
- Ensure `TrainingArguments` are set correctly for the environment.

#### [MODIFY] Part 7: Catastrophic Forgetting
- Add code to generate text using a "modern" prompt (e.g., "The capital of France is") for all 3 models.

#### [MODIFY] Part 8: Ablation Study
- The loop is provided, but I will ensure it runs correctly and captures the results in the `results` dictionary.

#### [MODIFY] Part 9: Final Report
- I will fill in the markdown tables with the results obtained from execution (or placeholders if execution is not performed by me).

## Verification Plan

### Automated Tests
- **Notebook Execution**: The primary verification is running the notebook cells.
    - **Data Prep**: Verify `lm_datasets` has correct shapes.
    - **Training**: Verify loss decreases and PPL is calculated.
    - **Generations**: Verify text is generated and looks "Shakespearean" for FT models.

### Manual Verification
- Review the generated text to ensure it makes sense.
- Check the PPL values:
    - Baseline should be high (it's not trained on this data).
    - Full FT should be low (it overfits/learns the style well).
    - LoRA should be competitive with Full FT but with far fewer parameters.
