import numpy as np
import matplotlib.pyplot as plt

# ====================== CORDIS Dominance from COVend Project ======================
# Real normalized dominance weights from Horizon Europe project COVend (ID 101045956)
dominance = np.array([0.510185, 0.323927, 0.013589, 0.152299])
probs_input = dominance / dominance.sum()

# ====================== Dirac-Solow-Swan Hamiltonian ======================
# Simplified diagonal Hamiltonian using dominance as coefficients
omega = probs_input
H_dirac = np.diag(omega)                     # Diagonal Dirac potential

# ====================== Time Evolution ======================
t = np.linspace(0, 5, 200)
capital_prob = []
state_t = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # Start with Academia-dominant state

for ti in t:
    U_t = np.exp(-1j * H_dirac * ti)
    state_t = U_t @ state_t
    state_t /= np.linalg.norm(state_t)
    capital_prob.append(np.abs(state_t[0])**2)   # Probability of disruptive capital (Academia component)

# ====================== Plot ======================
plt.figure(figsize=(10, 6))
plt.plot(t, capital_prob, linewidth=2.5, color='#1f77b4', label='Disruptive Capital (Academia)')
plt.xlabel('Time (arbitrary units)', fontsize=12)
plt.ylabel('Probability of Disruptive Capital', fontsize=12)
plt.title('Dirac-Solow-Swan Evolution from CORDIS Quadruple-Helix Data\n(project COVend, ID 101045956)', fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
