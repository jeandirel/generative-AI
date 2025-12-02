import json
import re

file_path = r"c:\Users\mon pc\Downloads\PGE5\Generative AI\llm_finetune (1).ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

results = {
    "baseline_ppl": "N/A",
    "full_ft_ppl": "N/A",
    "lora_ppl": "N/A",
    "generations": [],
    "ablation": "N/A",
    "forgetting": []
}

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check outputs
        outputs = cell.get('outputs', [])
        source = "".join(cell['source'])
        
        text_output = ""
        for out in outputs:
            if 'text' in out:
                text_output += "".join(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                text_output += "".join(out['data']['text/plain'])
            elif 'name' in out and out['name'] == 'stdout':
                text_output += "".join(out['text'])

        # Extract PPLs
        if "Baseline validation PPL:" in text_output:
            results["baseline_ppl"] = text_output.split("Baseline validation PPL:")[1].split()[0]
        if "Full FT validation PPL:" in text_output:
            results["full_ft_ppl"] = text_output.split("Full FT validation PPL:")[1].split()[0]
        if "LoRA validation PPL:" in text_output:
            results["lora_ppl"] = text_output.split("LoRA validation PPL:")[1].split()[0]
            
        # Extract Generations (heuristic)
        if "=== Baseline ===" in text_output:
            results["generations"].append(text_output)
            
        # Extract Ablation
        if "=== FINAL RESULTS ===" in text_output:
            results["ablation"] = text_output.split("=== FINAL RESULTS ===")[1]

        # Extract Forgetting
        if "=== Forgetting Test: Modern Prompts ===" in text_output:
             results["forgetting"].append(text_output)

print("Extraction Complete")
print(json.dumps(results, indent=2))
