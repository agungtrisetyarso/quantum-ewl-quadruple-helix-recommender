import matplotlib.pyplot as plt
import numpy as np

# ====================== DATA FROM TABLE III ======================
methods = ['MILP (exact)', 'GA', 'Tabu', 'SA', 'SQA', 'QAOA (p=2)']

# Optimality Gap (%)
gaps = [0.0, 1.84, 0.44, 0.36, 0.22, 1.53]

# Wall-clock time (seconds)
times = [87, 19, 14, 52, 7.8, 22]

# Feasibility (all are "yes" as per paper)
feasible = ['yes'] * len(methods)

# Colors (quantum-inspired highlighted)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
edge_color = 'black'

# ====================== PLOTTING FIGURE 2 ======================
fig, ax1 = plt.subplots(figsize=(11, 7))

x = np.arange(len(methods))
width = 0.35

# Left axis: Optimality Gap (%)
bars1 = ax1.bar(x - width/2, gaps, width, label='Optimality Gap (%)', 
                color=colors, edgecolor=edge_color, alpha=0.85)
ax1.set_ylabel('Optimality Gap (%)', fontsize=12, color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_ylim(0, max(gaps)*1.15)

# Right axis: Runtime
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, times, width, label='Wall-clock Time (s)', 
                color=colors, edgecolor=edge_color, alpha=0.65, hatch='//')
ax2.set_ylabel('Wall-clock Time (seconds)', fontsize=12, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

# Labels and title
ax1.set_xlabel('Solver', fontsize=12)
ax1.set_title('Figure 2. Solver Performance on Tractable Sub-instances (N\' = 64)', 
              fontsize=14, pad=20)

ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='#1f77b4')

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}', ha='center', va='bottom', fontsize=10, color='darkred')

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Grid and layout
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# Optional: Save high-res version
plt.savefig('Figure_2_Solver_Performance.png', dpi=300, bbox_inches='tight')
plt.show()
