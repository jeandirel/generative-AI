import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Baseline', 'Full FT', 'LoRA (r=16)']
ppls = [68.46, 25.64, 49.00]
colors = ['gray', 'green', 'blue']

# 1. Perplexity Comparison Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(models, ppls, color=colors)
plt.title('Validation Perplexity Comparison (Lower is Better)')
plt.ylabel('Perplexity')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom', fontweight='bold')

plt.savefig('ppl_comparison.png', dpi=300)
plt.close()

# 2. Ablation Study Curve
ranks = [1, 8, 16, 64]
ablation_ppls = [53.25, 53.61, 49.00, 53.53] 

plt.figure(figsize=(8, 5))
plt.plot(ranks, ablation_ppls, marker='o', linestyle='-', color='purple', linewidth=2)
plt.title('LoRA Ablation Study: Rank vs Perplexity')
plt.xlabel('LoRA Rank (r)')
plt.ylabel('Validation Perplexity')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(ranks)

# Annotate points
for r, p in zip(ranks, ablation_ppls):
    plt.text(r, p + 0.5, f"{p:.2f}", ha='center', va='bottom')

plt.savefig('ablation_curve.png', dpi=300)
plt.close()

print("Plots generated successfully.")
