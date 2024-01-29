from qutip import basis, tensor, qeye, Qobj, sesolve, destroy, create, num, mesolve,sigmax, sigmay, sigmaz
import numpy as np
import matplotlib.pyplot as plt

# Define the Hamiltonian components
def NV_Hamiltonian(B, D, S_g, S_d, N):
    Sx = sigmax()
    Sy = sigmay()
    Sz = sigmaz()

    # Define the Hamiltonian
    H_Zeeman = S_g * (B[0] * Sx + B[1] * Sy + B[2] * Sz)
    H_ZFS = D * Sz**2  # Assuming D is aligned along the z-axis for simplicity

    # Total Hamiltonian
    H = H_Zeeman + H_ZFS

def H_laserNV(Omega):
    # Define basis states for the ground triplet, excited triplet, and singlet states
    g0 = basis(3, 0)  # Ground state |g, 0>
    gp1 = basis(3, 1)  # Ground state |g, +1>
    gm1 = basis(3, 2)  # Ground state |g, -1>
    e0 = basis(3, 0)  # Excited state |e, 0>
    ep1 = basis(3, 1)  # Excited state |e, +1>
    em1 = basis(3, 2)  # Excited state |e, -1>
    s = basis(3, 0)  # Singlet state |s>

    # Define the optical pumping term for the transition |g,0> to |e,0>
    Omega = 2 * np.pi * 1e6  # Rabi frequency for the optical pumping (in rad/s)
    H_pump = Omega * (tensor(g0 * e0.dag(), qeye(3)) + tensor(e0 * g0.dag(), qeye(3)))

    # Define the non-radiative decay term for the transitions |e,±1> to |g,0> through |s>
    Gamma = 2 * np.pi * 1e3  # Decay rate (in rad/s)
    H_decay = Gamma * (tensor(gp1 * s.dag(), qeye(3)) + tensor(s * gp1.dag(), qeye(3)) +
                    tensor(gm1 * s.dag(), qeye(3)) + tensor(s * gm1.dag(), qeye(3)))
    return H_pump + H_decay