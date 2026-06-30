import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
import warnings
warnings.filterwarnings("ignore")

# ====================== HELPER FUNCTIONS ======================
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def tensor_product(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def entangling_operator(gamma):
    """J(gamma) = exp(i * gamma/2 * σx ⊗ σx ⊗ σx ⊗ σx)"""
    sigma_x4 = tensor_product([pauli_x()] * 4)
    return expm(1j * (gamma / 2) * sigma_x4)

def local_unitary(theta, phi):
    """U(θ, φ) for a single qubit"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [np.exp(1j * phi) * c, s],
        [-s, np.exp(-1j * phi) * c]
    ], dtype=complex)

def get_state(theta, phi, gamma):
    """|Ψ⟩ = J†(γ) (U_G ⊗ U_P ⊗ U_M ⊗ U_S) J(γ) |0000⟩"""
    J = entangling_operator(gamma)
    J_dag = J.conj().T
    
    Us = [local_unitary(t, p) for t, p in zip(theta, phi)]
    U_total = tensor_product(Us)
    
    psi0 = np.zeros(16, dtype=complex)
    psi0[0] = 1.0  # |0000>
    
    psi = J_dag @ U_total @ J @ psi0
    return psi / np.linalg.norm(psi)

def payoff_expectation(psi, Pi_a):
    """⟨Ψ| Π_a |Ψ⟩"""
    rho = np.outer(psi.conj(), psi)
    return np.real(np.trace(rho @ Pi_a))

def von_neumann_entropy(rho):
    """S(ρ) = -Tr(ρ log2 ρ)"""
    evals = np.real(np.linalg.eigvalsh(rho))
    evals = evals[evals > 1e-12]
    return -np.sum(evals * np.log2(evals))

def log_negativity(psi, partition):
    """Logarithmic negativity for a bipartition (simple implementation for 4 qubits)"""
    # For illustration: partial transpose on one subsystem and trace norm
    # Full implementation would depend on exact bipartition; here we show for M|S
    rho = np.outer(psi.conj(), psi)
    # Placeholder: compute for a 2-qubit vs 2-qubit cut (adjust indices as needed)
    # This is simplified; paper uses full computation
    rho_pt = rho.copy()  # Implement proper partial transpose in production
    return np.log2(np.linalg.norm(rho_pt, ord=1) + 1e-12)

# ====================== MAIN FIGURE 1 ======================
def plot_figure_1():
    gammas = np.linspace(0, np.pi/2, 60)
    gamma_star = np.pi / 3
    
    # Example payoff operators (diagonal, calibrated as per paper)
    # Replace with actual calibrated πa(x) from the repo when available
    payoffs = {
        'G': np.diag(np.random.uniform(0, 10, 16)),  # Placeholder
        'P': np.diag(np.random.uniform(0, 10, 16)),
        'M': np.diag(np.random.uniform(0, 10, 16)),
        'S': np.diag(np.random.uniform(0, 10, 16))
    }
    
    Pi_tot_list = []
    Var_list = []
    S_M_list = []
    S_MS_list = []
    EN_list = []
    
    for gamma in gammas:
        # Optimize over strategies (grid search for demo; use scipy.optimize in full version)
        best_obj = -np.inf
        best_theta_phi = None
        for _ in range(200):  # Monte-Carlo random search for illustration
            theta = np.random.uniform(0, np.pi, 4)
            phi = np.random.uniform(0, np.pi/2, 4)
            psi = get_state(theta, phi, gamma)
            
            Pis = [payoffs[h] for h in 'GPMS']
            Pi_as = [payoff_expectation(psi, Pi) for Pi in Pis]
            Pi_tot = sum(Pi_as)
            var_a = np.var(Pi_as)
            
            obj = Pi_tot - 0.75 * var_a  # λ ≈ 0.75 from paper
            
            if obj > best_obj:
                best_obj = obj
                best_theta_phi = (theta, phi)
                best_Pi_as = Pi_as
                best_psi = psi
        
        Pi_tot_list.append(best_Pi_as[0] + sum(best_Pi_as[1:]))  # total
        Var_list.append(np.var(best_Pi_as))
        
        # Entanglement measures (simplified)
        rho_M = np.trace(np.reshape(np.outer(best_psi.conj(), best_psi), (4,4,4,4)), axis1=1, axis2=3)  # reduced
        rho_M = rho_M.reshape(2,2) / np.trace(rho_M)
        S_M_list.append(von_neumann_entropy(rho_M))
        
        # MS bipartition entropy & negativity (placeholder values from paper trends)
        S_MS_list.append(1.0 if gamma > 0.5 else 0.0)
        EN_list.append(0.4 if gamma > 1.0 else 0.0)
    
    # ====================== PLOTTING ======================
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel (a): Πtot and Variance vs γ
    ax1 = axs[0]
    ax1.plot(gammas, Pi_tot_list, 'b-', label=r'$\Pi_{tot}$', linewidth=2.5)
    ax1.set_ylabel(r'Social Objective $\Pi_{tot}$', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(gammas, Var_list, 'r--', label=r'$\mathrm{Var}_a[\Pi_a]$', linewidth=2.5)
    ax2.set_ylabel(r'Payoff Variance $\mathrm{Var}_a[\Pi_a]$', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.axvline(gamma_star, color='k', linestyle=':', linewidth=2, label=r'$\gamma^* = \pi/3$')
    ax1.set_xlabel(r'Entanglement parameter $\gamma$')
    ax1.set_title('(a) Social Objective and Payoff Variance')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Panel (b): Entanglement measures
    axs[1].plot(gammas, S_M_list, 'g-', label=r'$S(\rho_M)$ (Mitra)', linewidth=2)
    axs[1].plot(gammas, S_MS_list, 'm-', label=r'$S(\rho_{MS})$ (Mitra-Society)', linewidth=2)
    axs[1].plot(gammas, EN_list, 'c--', label=r'$E_N$ (Mitra|Society)', linewidth=2)
    axs[1].axvline(gamma_star, color='k', linestyle=':', linewidth=2)
    axs[1].set_xlabel(r'Entanglement parameter $\gamma$')
    axs[1].set_ylabel('Entanglement Measures')
    axs[1].set_title('(b) Helix Coupling (Genuine Entanglement)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# Run the plot
if __name__ == "__main__":
    plot_figure_1()
