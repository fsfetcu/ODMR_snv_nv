import qutip as qt  
import numpy as np

# TODO:
#       - fix memory management
#       - add pulsed odmr


    

def get_vector_cartesian_qutip(A, theta, phi):
    """ 
    Compute cartesian coordinates of a vector from its spherical coordinates:
    norm A, polar angle theta, azimutal angle phi
    """
    vec_array = np.array([A * np.sin(theta) * np.cos(phi), 
                 A * np.sin(theta) * np.sin(phi),
                 A * np.cos(theta)])
    vec = qt.Qobj(vec_array)
    return vec

def get_vector_spherical_qutip(Avec):
    """ 
    Compute spherical coordinates of a vector from its cartesian coordinates.
    Avec is a qutip.Qobj representing the vector.
    """
    Avec_array = Avec.full().flatten()  # Convert Qobj to numpy array
    A0 = np.sqrt(np.dot(Avec_array, Avec_array))
    theta = np.arccos(Avec_array[2] / A0)
    
    phi = 0.0
    if Avec_array[0] != 0:
        phi = np.arctan(Avec_array[1] / Avec_array[0])
    if np.isnan(phi):
        phi = 0.
    if Avec_array[0] < 0:
        phi += np.pi

    return A0, theta, phi

def get_rotation_matrix_qutip(idx_nv):
    """ 
    Returns the transformation matrix from lab frame to the desired NV frame, 
    identified by idx_nv (can be 1, 2, 3, or 4).
    """
    if idx_nv == 1:
        RNV = qt.Qobj([[1/np.sqrt(6), -1/np.sqrt(6), -2/np.sqrt(6)],
                       [1/np.sqrt(2),  1/np.sqrt(2),  0],
                       [1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)]])
    elif idx_nv == 2:
        RNV = qt.Qobj([[-1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
                       [-1/np.sqrt(2), -1/np.sqrt(2),  0],
                       [-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)]])
    elif idx_nv == 3:
        RNV = qt.Qobj([[-1/np.sqrt(6), -1/np.sqrt(6),  2/np.sqrt(6)],
                       [-1/np.sqrt(2),  1/np.sqrt(2),  0],
                       [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
    elif idx_nv == 4:
        RNV = qt.Qobj([[1/np.sqrt(6),  1/np.sqrt(6),  2/np.sqrt(6)],
                       [1/np.sqrt(2), -1/np.sqrt(2),  0],
                       [1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)]])
    else:
        raise ValueError('Invalid index of NV orientation')
    
    return RNV

def transform_vector_lab_to_NV_frame_qutip(vec_in_lab, nv_idx=1):
    """ 
    Vector coordinates transformation from lab frame to desired NV frame.
    nv_idx can be 1, 2, 3 or 4.
    vec_in_lab is a qutip.Qobj representing the vector.
    """
    RNV = get_rotation_matrix_qutip(nv_idx)
    vec_in_nv = RNV * vec_in_lab
    return vec_in_nv

def transform_vector_NV_to_lab_frame_qutip(vec_in_nv, nv_idx=1):
    """ 
    Vector coordinates transformation from given NV frame to lab frame.
    nv_idx can be 1, 2, 3, or 4.
    vec_in_nv is a qutip.Qobj representing the vector.
    """
    RNV = get_rotation_matrix_qutip(nv_idx)
    vec_in_lab = RNV.dag() * vec_in_nv
    return vec_in_lab

def transform_all_frames_qutip(B0, theta, phi):
    """ 
    Compute cartesian coordinates of a vector in all 4 NV frames, 
    based on its spherical coordinates in lab frame.
    """
    Bvec = get_vector_cartesian_qutip(B0, theta, phi)
        
    # Ensure that the transformation functions return 3D vectors
    Bvec_list = [transform_vector_lab_to_NV_frame_qutip(Bvec, idx).full().flatten() 
                 for idx in range(1, 5)]
  
    # Converting the result to a more appropriate format (like a list or an array)
    Bvec_list = [vec.tolist() for vec in Bvec_list]

    return Bvec_list

# Spherical coordiantes transformation
def transform_spherical_nv_to_lab_frame_qutip(theta_nv, phi_nv, idx_nv=1):
    """ 
    Spherical coordinates transformation from given NV frame to lab frame.
    nv_idx can be 1, 2, 3 or 4.
    """
    vec_in_nv = get_vector_cartesian_qutip(1, theta_nv, phi_nv)
    vec_in_lab = transform_vector_NV_to_lab_frame_qutip(vec_in_nv, idx_nv)
    _, theta_lab, phi_lab = get_vector_spherical_qutip(vec_in_lab)
    return theta_lab, phi_lab

def transform_spherical_lab_to_nv_frame_qutip(theta_lab, phi_lab, idx_nv=1):
    """ 
    Spherical coordinates transformation from lab frame to given NV frame.
    nv_idx can be 1, 2, 3 or 4.
    """
    vec_in_lab = get_vector_cartesian_qutip(1, theta_lab, phi_lab)
    vec_in_nv = transform_vector_lab_to_NV_frame_qutip(vec_in_lab, idx_nv)
    _, theta_nv, phi_nv = get_vector_spherical_qutip(vec_in_nv)
    return theta_nv, phi_nv

# =============================================================================
# Single NV center Hamiltonian
# =============================================================================
# Constants 
# NV fine and hyperfine constants (in MHz)
# Constants remain the same as they are just numerical values


D_0 = 2.87e3
Apar = -2.14
Aperp = -2.7
PQ = -4.95

# Magnetic coupling constants (in SI units)
muB = 9.274e-24
gNV = 2.0028
muN = 5.051e-27
gN = 0.404
h = 6.626e-34

# Gyromagnetic ratios (in MHz/G)
gammaNV = muB * gNV / h / 1e10 # NV gyromagnetic ratio 
gammaN = muN * gN / h / 1e10 # N gyromagnetic ratio (in MHz/G)

# Electric coupling constants
d_parallel = 3.5e-9 # MHz/(V/m)
d_transverse = 0.17e-6 # MHz/(V/m)

# Pauli matrices using QuTiP
S_x = qt.Qobj(1 / np.sqrt(2) * np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]]))
S_y = qt.Qobj(1 / np.sqrt(2) * 1j * np.array([[0, 1, 0],
                                              [-1, 0, 1],
                                              [0, -1, 0]]))
S_z = qt.Qobj(np.array([[1, 0, 0],
                        [0, 0, 0], 
                        [0, 0, -1]]))
SI = qt.qeye(3)

# Matrix useful for definition of Hamiltonian
S_zfs = S_z * S_z - 2/3 * SI

def NV_transitionsElevels_qutip(B, E):
    """
    Input: magnetic field and electric field, defined in NV center frame
    Output: This function diagonalizes the Hamiltonian Hgs and returns 9 eigenenergies and its vectors
    """

    # Hamiltonian
    # Fine and hyperfine terms
    HZFS = D_0 * qt.tensor(S_zfs, SI)  # Zero-field splitting
    HHFPar = Apar * qt.tensor(S_z, S_z)  # Axial hyperfine interaction
    HHFPerp = Aperp * (qt.tensor(S_x, S_x) + qt.tensor(S_y, S_y))  # Non-axial hyperfine interaction
    HNucQ = PQ * qt.tensor(SI, S_zfs)  # Nuclear quadrupole interaction
    # print(f"this is SI: {S_x.shape}")
    # print(f"this is{B[0].shape}")
    # Magnetic field coupling terms
    HBEl = gammaNV * qt.tensor(B[0] * S_x + B[1] * S_y + B[2] * S_z, SI)  # Electric Zeeman coupling
    HBNuc = gammaN * qt.tensor(SI, B[0] * S_x + B[1] * S_y + B[2] * S_z)  # Nuclear Zeeman coupling

    # Electric field coupling terms
    H_elec = (E[2] * d_parallel * qt.tensor(S_zfs, SI)
              + E[0] * d_transverse * qt.tensor((S_y * S_y - S_x * S_x), SI)
              + E[1] * d_transverse * qt.tensor((S_x * S_y + S_y * S_x), SI))

    H_total = HZFS + HBEl + HBNuc + H_elec + HHFPar + HHFPerp + HNucQ
    E_I, vec_I = H_total.eigenstates()

    return E_I, vec_I

def NV_GS_Hamiltonian_MWprobe_qutip(Bmw):
    """
    Compute interaction Hamiltonian, with MW vector Bmw defined in NV center frame.
    """

    # Magnetic field coupling terms
    HintEl = gammaNV * qt.tensor(Bmw[0] * S_x + Bmw[1] * S_y + Bmw[2] * S_z, SI)  # To electric spin
    HintNuc = gammaN * qt.tensor(SI, Bmw[0] * S_x + Bmw[1] * S_y + Bmw[2] * S_z)  # To nuclear spin

    # Total interaction Hamiltonian
    Hint = HintEl + HintNuc
    return Hint


# =============================================================================
# Computation of ODMR spectrum
# =============================================================================
def lorentzian_qutip(x, x0, fwhm):
    return 1 / (1 + (x - x0)**2 / (fwhm / 2)**2)

def ESR_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth):
    """
    All vectors are defined in NV frame.
    Calculates ESR transition strengths for a single NV center.
    """
    nMW = len(MWfreq)  # Number of frequency points
    Tstrength = np.zeros(nMW)  # Transition strength

    E_I, vec_I = NV_transitionsElevels_qutip(Bvec, Evec)  # Eigenenergies and eigenvectors
    Hint = NV_GS_Hamiltonian_MWprobe_qutip(MWvec)  # Interaction Hamiltonian

    # Calculate transition strengths
    for initS in range(9):  # Sweep over all initial states
        initFreq = E_I[initS]  # Frequency
        initVec = vec_I[initS]

        for finS in range(initS, 9):  # Sweep over all final states
            finFreq = E_I[finS]  # Frequency
            finVec = vec_I[finS]  # State

            # Transition matrix element and transition amplitude
            TME = (finVec.dag() * Hint * initVec).data.toarray()[0,0]
            TA = np.abs(TME)**2

            # Add Lorentzian lineshape
            TS = TA * lorentzian_qutip(MWfreq, np.abs(finFreq - initFreq), Linewidth)
            
            Tstrength += TS

    return Tstrength


def ESR_pulsed_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth):
    """
    All vectors are defined in NV frame.
    Calculates ESR transition strengths for a single NV center using pulsed ESR.
    """

    pi_pulse_duration = np.pi / (2.87e9 * 2 * np.pi)

    nMW = len(MWfreq)  # Number of microwave frequency points
    Tstrength = np.zeros(nMW)  # Transition strength
    
    # Eigenenergies and eigenvectors
    E_I, vec_I = NV_transitionsElevels_qutip(Bvec, Evec)
    # Interaction Hamiltonian
    Hint = NV_GS_Hamiltonian_MWprobe_qutip(MWvec)
    
    # Define the time-dependent coefficient function for the pulse
    def pulse_coeff(t, args):
        return 1 if 0 <= t <= pi_pulse_duration else 0
    
    # Loop over MW frequencies
    for idx, mw_freq in enumerate(MWfreq):
        # Apply the π pulse for a duration of pi_pulse_duration
        H_pulse = [Hint, pulse_coeff]
        
        # Initial state of the system, typically the ground state |0>
        psi0 = qt.tensor(qt.basis(3, 0), qt.basis(3, 0))  # Replace with the correct initial state
        
        # Solve the time-dependent Schrödinger equation during the π pulse
        result = qt.mesolve(H_pulse, psi0, np.linspace(0, pi_pulse_duration, 100), [], [])
        
        # The state at the end of the π pulse
        psi_end = result.states[-1]
        
        # Calculate transition strengths
        for finS in range(len(vec_I)):  # Sweep over all final states
            finFreq = E_I[finS]  # Frequency of the final state
            finVec = vec_I[finS]  # State vector of the final state
            
            # Transition matrix element and transition amplitude
           
            TME = (finVec.dag() * psi_end).data.toarray()[0,0]
            TA = np.abs(TME)**2
            
            # Apply Lorentzian lineshape
            TS = TA * lorentzian_qutip(mw_freq, np.abs(finFreq - E_I[0]), Linewidth)
            Tstrength[idx] += TS
            
    return Tstrength

def ESR_NVensemble_qutip(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    """
    All vectors are defined in lab frame (spherical coordinates).
    Calculates ESR transition strengths for an NV ensemble.
    """
    nMW = len(MWfreq)  # Number of frequency points
    Tstrength = np.zeros(nMW)  # Transition strength

    Bvector_list = transform_all_frames_qutip(B0, thetaB, phiB)
    Evector_list = transform_all_frames_qutip(E0, thetaE, phiE)
    MWvector_list = transform_all_frames_qutip(1, thetaMW, phiMW)

    for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
        Tstrength += ESR_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth)

    n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
    return Tstrength / n_NV


def pulsed_ESR_NVensemble_qutip(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    """
    All vectors are defined in lab frame (spherical coordinates).
    Calculates ESR transition strengths for an NV ensemble.
    """
    nMW = len(MWfreq)  # Number of frequency points
    Tstrength = np.zeros(nMW)  # Transition strength

    Bvector_list = transform_all_frames_qutip(B0, thetaB, phiB)
    Evector_list = transform_all_frames_qutip(E0, thetaE, phiE)
    MWvector_list = transform_all_frames_qutip(1, thetaMW, phiMW)

    for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
        Tstrength += ESR_pulsed_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth)

    n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
    return Tstrength / n_NV



def ESR_VNensemble_qutip(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    return ESR_NVensemble_qutip(MWfreq, thetaMW + np.pi, phiMW, 
                                B0, thetaB + np.pi, phiB, 
                                E0, thetaE + np.pi, phiE, Linewidth)

def ESR_NV_VN_ensemble_qutip(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    T_NV = ESR_NVensemble_qutip(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
    T_VN = ESR_VNensemble_qutip(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth)
    return (T_NV + T_VN) / 2

def add_noise_qutip(Tstrength):
    """Add 'random' noise to the Transition Strengths of a single NV center"""
    noise_level = 0.02 * np.max(Tstrength)
    # Add Gaussian noise to each transition strength value
    noisy_Tstrength = np.random.normal(0, noise_level, Tstrength.shape)
    # Ensure that the noisy transition strength does not go negative
    # noisy_Tstrength = np.maximum(noisy_Tstrength, 0)
    return noisy_Tstrength


def ESR_NVensemble_qutip_noisy(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    """
    All vectors are defined in lab frame (spherical coordinates).
    Calculates ESR transition strengths for an NV ensemble.
    Only difference with ESR_NVensemble_qutip(...) is the randmonly added (thermal) noise.
    """
    nMW = len(MWfreq)  # Number of frequency points
    Tstrength = np.zeros(nMW)  # Transition strength

    Bvector_list = transform_all_frames_qutip(B0, thetaB, phiB)
    Evector_list = transform_all_frames_qutip(E0, thetaE, phiE)
    MWvector_list = transform_all_frames_qutip(1, thetaMW, phiMW)

    for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
        Tstrength = Tstrength + ESR_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth) + add_noise_qutip(Tstrength)

    n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
    return Tstrength / n_NV

def pulsed_ESR_NVensemble_qutip_noisy(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
    """
    All vectors are defined in lab frame (spherical coordinates).
    Calculates ESR transition strengths for an NV ensemble.
    """
    nMW = len(MWfreq)  # Number of frequency points
    Tstrength = np.zeros(nMW)  # Transition strength

    Bvector_list = transform_all_frames_qutip(B0, thetaB, phiB)
    Evector_list = transform_all_frames_qutip(E0, thetaE, phiE)
    MWvector_list = transform_all_frames_qutip(1, thetaMW, phiMW)

    for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
        Tstrength = Tstrength + ESR_pulsed_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth) + add_noise_qutip(Tstrength)

    n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
    return Tstrength / n_NV


# =============================================================================
# Computation of Lockin ODMR spectrum
# =============================================================================
def dispersive_lineshape_qutip(x, x0, fwhm):
    xred = (x - x0) / (fwhm / 2)
    return -2 * xred / (1 + xred**2)**2

def ESR_Lockin_singleNV_qutip(MWfreq, MWvec, Bvec, Evec, Linewidth):
    """
    All vectors are defined in NV frame.
    Calculates ESR lock-in transition strengths for a single NV center.
    """
    nMW = len(MWfreq)  # Number of frequency points
    Lockin = np.zeros(nMW)  # Transition strength

    E_I, vec_I = NV_transitionsElevels_qutip(Bvec, Evec)  # Eigenenergies and eigenvectors
    Hint = NV_GS_Hamiltonian_MWprobe_qutip(MWvec)  # Interaction Hamiltonian

    # Calculate transition strengths
    for initS in range(9):  # Sweep over all initial states
        initFreq = E_I[initS]  # Frequency
        initVec = vec_I[initS]

        for finS in range(initS, 9):  # Sweep over all final states
            finFreq = E_I[finS]  # Frequency
            finVec = vec_I[finS]  # State

            # Transition matrix element and transition amplitude
            TME = (finVec.dag() * Hint * initVec).data.toarray()[0,0]
            TA = np.abs(TME)**2

            # Add dispersive lineshape
            TS = TA * dispersive_lineshape_qutip(MWfreq, np.abs(finFreq - initFreq), Linewidth)

            Lockin += TS

    return Lockin


