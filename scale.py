import matplotlib.pyplot as plt
import numpy as np

# ====================== DATA FROM TABLE IV ======================
methods = ['GA', 'Tabu', 'SA', 'SQA']

# Mean best objective (lower = better)
means = [-1_248_300, -1_251_100, -1_249_800, -1_252_400]

# Standard deviation across 5 seeds
stds = [4_200, 3_100, 2_800, 2_100]

# ====================== PLOTTING FIGURE 3 ======================
fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(len(methods))
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # GA, Tabu, SA, SQA

bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, 
              edgecolor='black', alpha=0.85, width=0.6)

# Labels and title
ax.set_xlabel('Solver', fontsize=12)
ax.set_ylabel('Best QUBO Objective Value\n(lower is better)', fontsize=12)
ax.set_title('Figure 3. Full-scale Solution Quality (N = 13,603 stations)', 
             fontsize=14, pad=20)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

# Value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 800, 
            f'{height:,.0f}', ha='center', va='top', 
            fontsize=10, color='white', fontweight='bold')

# Highlight SQA as best
ax.text(3, means[3] - 1500, 'Best (SQA)', ha='center', fontsize=11, 
        color='#9467bd', fontweight='bold')

# Add note about improvement
ax.annotate('SQA outperforms GA by ≈0.33%', xy=(0, means[0]), 
            xytext=(1.5, means[0]-800), arrowprops=dict(arrowstyle='->'),
            fontsize=10, ha='center')

ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Save high-resolution version
plt.savefig('Figure_3_Full_Scale_Solution_Quality.png', dpi=300, bbox_inches='tight')
plt.show()
