import matplotlib.pyplot as plt
import numpy as np

# ====================== DATA FROM TABLE IV ======================
methods = ['GA', 'Tabu', 'SA', 'SQA']

# Mean wall-clock time (seconds)
means = [980, 1150, 1320, 1410]

# Standard deviation
stds = [120, 90, 150, 110]

# ====================== PLOTTING FIGURE 4 ======================
fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(len(methods))
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # GA, Tabu, SA, SQA

bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, 
              edgecolor='black', alpha=0.85, width=0.6)

# Labels and title
ax.set_xlabel('Solver', fontsize=12)
ax.set_ylabel('Wall-clock Runtime (seconds)', fontsize=12)
ax.set_title('Figure 4. Full-scale Wall-clock Runtime\n(N = 13,603 stations, 5 seeds)', 
             fontsize=14, pad=20)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 30,
            f'{height:.0f} ± {stds[i]:.0f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add annotation
ax.annotate('All methods < 25 minutes', 
            xy=(2.5, 1200), xytext=(1.5, 1600),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Save high-resolution version
plt.savefig('Figure_4_Full_Scale_Runtime.png', dpi=300, bbox_inches='tight')
plt.show()
