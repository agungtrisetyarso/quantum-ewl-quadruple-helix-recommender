import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.quantum_info import hellinger_fidelity

# ====================== 1. COVend Parameters ======================
p = np.array([0.5102, 0.3239, 0.0136, 0.1523])
theta = 2 * np.arcsin(np.sqrt(p))
print("θ angles (rad):", np.round(theta, 6))

# ====================== 2. Exact 4-qubit EWL Circuit ======================
def create_covend_ewl_circuit():
    qc = QuantumCircuit(4, 4)
    qc.h(range(4))
    qc.s(range(4))
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    for i in range(4):
        qc.ry(theta[i], i)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.sdg(range(4))
    qc.measure(range(4), range(4))
    return qc

qc = create_covend_ewl_circuit()
print("Original circuit depth :", qc.depth())

# ====================== 3. Ideal Simulator ======================
sim = AerSimulator()
sim_counts = sim.run(qc, shots=8192).result().get_counts()

# ====================== 4. Run on IBM Quantum Hardware ======================
service = QiskitRuntimeService(
    channel="ibm_quantum_platform", 
    token="1vzUcO6Rd4IWvxbw3H08sVigGZXlrS_4aYluY1KCQQRH",
    instance="open-instance"
)

backend_name = "ibm_kingston"
backend = service.backend(backend_name)

# Transpile to native gates
qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
print("Transpiled circuit depth:", qc_transpiled.depth())

sampler = SamplerV2(mode=backend)
job = sampler.run([qc_transpiled], shots=8192)
print("✅ Job submitted to", backend_name)
print("Job ID:", job.job_id())

# Wait for result
result = job.result()

# Correct way for latest SamplerV2 (classical register is named "c")
hw_counts = result[0].data.c.get_counts()

# ====================== 5. Compute Fidelity ======================
fidelity = hellinger_fidelity(sim_counts, hw_counts)
print(f"\n🎯 HARDWARE FIDELITY on {backend_name}: {fidelity:.4f}")
