import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Reuse your existing dominance weights and Hamiltonian builder
p = np.array([0.5102, 0.3239, 0.0136, 0.1523])
q = np.array([0.68, 0.55, 0.72, 0.61])   # approximate post-game scores from Table III

def build_hamiltonian(q, lambda_c=0.2, beta=1.0, omega0=0.5):
    n = 4
    H = np.diag(omega0 + beta * q)
    Gamma = np.array([[0,0.45,0.25,0.35],
                      [0.45,0,0.30,0.40],
                      [0.25,0.30,0,0.20],
                      [0.35,0.40,0.20,0]])
    for i in range(n):
        for j in range(i+1,n):
            H[i,j] = lambda_c * Gamma[i,j]
            H[j,i] = H[i,j]
    return H

# Generate phase diagram data
lambda_vals = np.linspace(0, 0.6, 60)
beta_vals   = np.linspace(0, 1.5, 60)
P_disruptive = np.zeros((len(lambda_vals), len(beta_vals)))

initial_state = np.array([1.0, 0, 0, 0], dtype=complex)
t_final = 10.0

for i, lam in enumerate(lambda_vals):
    for j, b in enumerate(beta_vals):
        H = build_hamiltonian(q, lambda_c=lam, beta=b)
        U = expm(-1j * H * t_final)
        psi_t = U @ initial_state
        P_disruptive[i,j] = np.abs(psi_t[0])**2

# Plot
plt.figure(figsize=(8,6))
plt.contourf(lambda_vals, beta_vals, P_disruptive.T, levels=50, cmap='RdYlBu_r')
plt.colorbar(label=r'$P_\text{disruptive}(t=10)$')
plt.contour(lambda_vals, beta_vals, P_disruptive.T, levels=[0.5], colors='white', linestyles='--')
plt.xlabel(r'Inter-helix coupling $\lambda$')
plt.ylabel(r'Innovation boost $\beta$')
plt.title('Phase diagram of disruptive vs sustaining capital regimes')
plt.tight_layout()
plt.savefig('phase_diagram_disruptive.pdf', dpi=300, bbox_inches='tight')
plt.show()
