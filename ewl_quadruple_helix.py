
#### Main file: `ewl_quadruple_helix.py` (copy the code below)

```python
"""
Parameterized 4-Qubit EWL Quantum Game Circuit
for Quadruple Helix Disruptive Innovation Recommender Systems

Authors: Agung Trisetyarso, Fithra Faisal Hastiadi
Date: May 2026
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# ====================== CORDIS Dominance from COVend Project ======================
# Real normalized dominance weights from Horizon Europe project COVend (ID 101045956)
dominance = np.array([0.510185, 0.323927, 0.013589, 0.152299])
dominance = dominance / dominance.sum()

labels = ['Academia', 'Industry', 'Government', 'Civil_Society']

print("=== CORDIS Dominance (normalized) ===")
for lbl, p in zip(labels, dominance):
    print(f"  {lbl}: {p:.4f}")

# Compute rotation angles for parameterized local strategies
theta = 2 * np.arcsin(np.sqrt(dominance))

# ====================== Build the 4-Qubit Parameterized EWL Circuit ======================
qc = QuantumCircuit(4, 4)

# Initial |++++⟩ state
qc.h([0, 1, 2, 3])

# EWL entangler Ĵ
qc.s([0, 1, 2, 3])
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# Parameterized local strategies U_i = R_y(θ_i)
for i in range(4):
    qc.ry(theta[i], i)

# Inverse entangler Ĵ†
qc.cx(2, 3)
qc.cx(1, 2)
qc.cx(0, 1)
qc.sdg([0, 1, 2, 3])

# Measurement
qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

print(f"\nCircuit built — Total gates: {qc.count_ops()}")
print(f"Circuit depth: {qc.depth()}")

# Text diagram
print("\nCircuit (text representation):")
print(qc.draw('text'))

# ====================== Simulation ======================
simulator = AerSimulator()
t_qc = transpile(qc, simulator)
job = simulator.run(t_qc, shots=8192)
result = job.result()
counts = result.get_counts()

print("\n=== Measurement Results (Recommender Scores) ===")
plot_histogram(counts, figsize=(14, 6),
               title="Parameterized EWL Quantum Game Recommender Scores\n"
                     "(Local strategies tuned by real CORDIS dominance)")
plt.show()
