#!/usr/bin/python3
# -*- coding: utf-8 -*- 

"""
@author:https://github.com/fsfetcu
"""

#TODO
# Update the docstrings

import numpy as np
from qutip import tensor, qeye, Qobj, sigmax, sigmay, sigmaz
from vectors import vec


class singleNVhamiltonian:
    """
    A module for the single NV Hamiltonian and related functions.
    """

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

    # Pauli matrices 
    S_x = Qobj(1 / np.sqrt(2) * np.array([[0, 1, 0],
                                            [1, 0, 1],
                                            [0, 1, 0]]))
    S_y = Qobj(1 / np.sqrt(2) * 1j * np.array([[0, 1, 0],
                                                [-1, 0, 1],
                                                [0, -1, 0]]))
    S_z = Qobj(np.array([[1, 0, 0],
                            [0, 0, 0], 
                            [0, 0, -1]]))
    SI = qeye(3)

    # Matrix useful for definition of Hamiltonian
    S_zfs = S_z * S_z - 2/3 * SI
    
    @staticmethod
    def NV_eigenEnergiesStates(B, E):
        """
        NV_transitionsElevels_qutip
        Input: magnetic field and electric field, defined in NV center frame
        Output: This function diagonalizes the Hamiltonian Hgs and returns 9 eigenenergies and its vectors

        Parameters
        ----------
        B : numpy.ndarray
            Magnetic field vector.
        E : numpy.ndarray
            Electric field vector.
        
        Returns
        -------
        numpy.ndarray
            Eigenenergies of the Hamiltonian.
        numpy.ndarray
            Eigenstates of the Hamiltonian.
        """

        # Fine and hyperfine terms
        HZFS = singleNVhamiltonian.D_0 * tensor(singleNVhamiltonian.S_zfs, singleNVhamiltonian.SI)  # Zero-field splitting
        HHFPar = singleNVhamiltonian.Apar * tensor(singleNVhamiltonian.S_z, singleNVhamiltonian.S_z)  # Axial hyperfine interaction
        HHFPerp = singleNVhamiltonian.Aperp * (tensor(singleNVhamiltonian.S_x, singleNVhamiltonian.S_x) + tensor(singleNVhamiltonian.S_y, singleNVhamiltonian.S_y))  # Non-axial hyperfine interaction
        HNucQ = singleNVhamiltonian.PQ * tensor(singleNVhamiltonian.SI, singleNVhamiltonian.S_zfs)  # Nuclear quadrupole interaction
        
        # Magnetic field coupling terms
        HBEl = singleNVhamiltonian.gammaNV * tensor(B[0] * singleNVhamiltonian.S_x + B[1] * singleNVhamiltonian.S_y + B[2] * singleNVhamiltonian.S_z, singleNVhamiltonian.SI)  # Electric Zeeman coupling
        HBNuc = singleNVhamiltonian.gammaN * tensor(singleNVhamiltonian.SI, B[0] * singleNVhamiltonian.S_x + B[1] * singleNVhamiltonian.S_y + B[2] * singleNVhamiltonian.S_z)  # Nuclear Zeeman coupling

        # Electric field coupling terms
        H_elec = (E[2] * singleNVhamiltonian.d_parallel * tensor(singleNVhamiltonian.S_zfs, singleNVhamiltonian.SI)
                + E[0] * singleNVhamiltonian.d_transverse * tensor((singleNVhamiltonian.S_y * singleNVhamiltonian.S_y - singleNVhamiltonian.S_x * singleNVhamiltonian.S_x), singleNVhamiltonian.SI)
                + E[1] * singleNVhamiltonian.d_transverse * tensor((singleNVhamiltonian.S_x * singleNVhamiltonian.S_y + singleNVhamiltonian.S_y * singleNVhamiltonian.S_x), singleNVhamiltonian.SI))

        H_total = HZFS + HBEl + HBNuc + H_elec + HHFPar + HHFPerp + HNucQ
        E_I, vec_I = H_total.eigenstates()

        return E_I, vec_I

    @staticmethod
    def GShamiltonianMWint(Bmw):
        """
        Compute interaction Hamiltonian, with MW vector Bmw defined in NV center frame,
        including detuning from the resonance frequency.

        Parameters
        ----------
        Bmw : numpy.ndarray
            Magnetic field vector.
        detuning : float
            Detuning from the resonance frequency (in Hz).

        Returns
        -------
        Qobj
            Interaction Hamiltonian.
        """
        # Magnetic field coupling terms
        HintEl = singleNVhamiltonian.gammaNV * tensor(Bmw[0] * singleNVhamiltonian.S_x + Bmw[1] * singleNVhamiltonian.S_y + Bmw[2] * singleNVhamiltonian.S_z, singleNVhamiltonian.SI)  # To electric spin
        HintNuc = singleNVhamiltonian.gammaN * tensor(singleNVhamiltonian.SI, Bmw[0] * singleNVhamiltonian.S_x + Bmw[1] * singleNVhamiltonian.S_y + Bmw[2] * singleNVhamiltonian.S_z)  # To nuclear spin
        
        # Include detuning
        
        # Total interaction Hamiltonian
        Hint = HintEl + HintNuc 
        return Hint

    @staticmethod
    def free_hamiltonian(B, E):
        """
        NV_free_Hamiltonian_qutip
        Compute free Hamiltonian, with B and E defined in NV center frame.

        Parameters
        ----------
        B : numpy.ndarray
        Magnetic field vector. 
        E : numpy.ndarray
        Electric field vector.

        Returns
        -------
        Qobj
        Free Hamiltonian.
        """
        # Fine and hyperfine terms
        HZFS = singleNVhamiltonian.D_0 * tensor(singleNVhamiltonian.S_zfs, singleNVhamiltonian.SI)  # Zero-field splitting
        HHFPar = singleNVhamiltonian.Apar * tensor(singleNVhamiltonian.S_z, singleNVhamiltonian.S_z)  # Axial hyperfine interaction
        HHFPerp = singleNVhamiltonian.Aperp * (tensor(singleNVhamiltonian.S_x, singleNVhamiltonian.S_x) + tensor(singleNVhamiltonian.S_y, singleNVhamiltonian.S_y))  # Non-axial hyperfine interaction
        HNucQ = singleNVhamiltonian.PQ * tensor(singleNVhamiltonian.SI, singleNVhamiltonian.S_zfs)  # Nuclear quadrupole interaction
        
        # Magnetic field coupling terms
        HBEl = singleNVhamiltonian.gammaNV * tensor(B[0] * singleNVhamiltonian.S_x + B[1] * singleNVhamiltonian.S_y + B[2] * singleNVhamiltonian.S_z, singleNVhamiltonian.SI)  # Electric Zeeman coupling
        HBNuc = singleNVhamiltonian.gammaN * tensor(singleNVhamiltonian.SI, B[0] * singleNVhamiltonian.S_x + B[1] * singleNVhamiltonian.S_y + B[2] * singleNVhamiltonian.S_z)  # Nuclear Zeeman coupling

        # Electric field coupling terms
        H_elec = (E[2] * singleNVhamiltonian.d_parallel * tensor(singleNVhamiltonian.S_zfs, singleNVhamiltonian.SI)
                + E[0] * singleNVhamiltonian.d_transverse * tensor((singleNVhamiltonian.S_y * singleNVhamiltonian.S_y - singleNVhamiltonian.S_x * singleNVhamiltonian.S_x), singleNVhamiltonian.SI)
                + E[1] * singleNVhamiltonian.d_transverse * tensor((singleNVhamiltonian.S_x * singleNVhamiltonian.S_y + singleNVhamiltonian.S_y * singleNVhamiltonian.S_x), singleNVhamiltonian.SI))

        H_total = HZFS + HBEl + HBNuc + H_elec + HHFPar + HHFPerp + HNucQ

        return H_total


    @staticmethod
    def simpleHamiltonEigen(B, E):
        """
        Compute eigenvalues and eigenvectors of the Hamiltonian, with B and E defined in NV center frame.

        Parameters
        ----------
        B : numpy.ndarray
            Magnetic field vector.
        E : numpy.ndarray
            Electric field vector.

        Returns
        -------
        numpy.ndarray
            Eigenvalues of the Hamiltonian.
        numpy.ndarray
            Eigenvectors of the Hamiltonian.
        """
         # Fine and hyperfine terms
        HZFS = singleNVhamiltonian.D_0 * singleNVhamiltonian.S_zfs  # Zero-field splitting
        
        
        # Magnetic field coupling terms
        HBEl = singleNVhamiltonian.gammaNV * (B[0] * singleNVhamiltonian.S_x + B[1] * singleNVhamiltonian.S_y + B[2] * singleNVhamiltonian.S_z) # Electric Zeeman coupling

        # Electric field coupling terms
        H_elec = (E[2] * singleNVhamiltonian.d_parallel * (singleNVhamiltonian.S_zfs) \
                + E[0] * singleNVhamiltonian.d_transverse * ((singleNVhamiltonian.S_y * singleNVhamiltonian.S_y - singleNVhamiltonian.S_x * singleNVhamiltonian.S_x))
                + E[1] * singleNVhamiltonian.d_transverse * tensor((singleNVhamiltonian.S_x * singleNVhamiltonian.S_y + singleNVhamiltonian.S_y * singleNVhamiltonian.S_x)))

        H_total = HZFS + HBEl + H_elec

        E_I, vec_I = H_total.eigenstates()

        return E_I, vec_I


    @staticmethod
    def simpleMWint(Bmw):
        """
        Compute interaction Hamiltonian, with MW vector Bmw defined in NV center frame,
        including detuning from the resonance frequency.

        Parameters
        ----------
        Bmw : numpy.ndarray
            Magnetic field vector.

        Returns
        -------
        Qobj
            Interaction Hamiltonian.
        """
        # Magnetic field coupling terms
        HintEl = singleNVhamiltonian.gammaNV * (Bmw[0] * singleNVhamiltonian.S_x + Bmw[1] * singleNVhamiltonian.S_y + Bmw[2] * singleNVhamiltonian.S_z)

        # Total interaction Hamiltonian
        Hint = HintEl
        return Hint


    @staticmethod
    def simpleFree(B, E):
        """
        Compute free Hamiltonian, with B and E defined in NV center frame.

        Parameters
        ----------
        B : numpy.ndarray
            Magnetic field vector.
        E : numpy.ndarray
            Electric field vector.

        Returns
        -------
        Qobj
            Free Hamiltonian.
        """
            # Fine and hyperfine terms
        HZFS = singleNVhamiltonian.D_0 * singleNVhamiltonian.S_zfs  # Zero-field splitting
        
        
        # Magnetic field coupling terms
        HBEl = singleNVhamiltonian.gammaNV * (B[0] * singleNVhamiltonian.S_x + B[1] * singleNVhamiltonian.S_y + B[2] * singleNVhamiltonian.S_z) # Electric Zeeman coupling

        # Electric field coupling terms
        H_elec = (E[2] * singleNVhamiltonian.d_parallel * (singleNVhamiltonian.S_zfs) \
                + E[0] * singleNVhamiltonian.d_transverse * ((singleNVhamiltonian.S_y * singleNVhamiltonian.S_y - singleNVhamiltonian.S_x * singleNVhamiltonian.S_x))
                + E[1] * singleNVhamiltonian.d_transverse * tensor((singleNVhamiltonian.S_x * singleNVhamiltonian.S_y + singleNVhamiltonian.S_y * singleNVhamiltonian.S_x)))

        H_total = HZFS + HBEl + H_elec


        return H_total


class SingleSnVHamiltonian:
    """
    A class for the single SnV Hamiltonian and related functions.
    """

    # Define Pauli matrices for S=1/2 system
    sigma_x = Qobj([[0, 1], [1, 0]])
    sigma_y = Qobj([[0, -1j], [1j, 0]])
    sigma_z = Qobj([[1, 0], [0, -1]])
    I = qeye(2)

    # Constants
    lambda_SO = 850e9  # Spin-orbit coupling constant in Hz
    gamma_e = 2 * 9.274e-24 / 6.626e-34  # Gyromagnetic ratio for electron

    Upsilon_x = 50e9/(2*np.pi)
    Upsilon_y = np.sqrt(177e9**2 - Upsilon_x**2)/(2*np.pi)
    delta_g = 0.014
    f_g = 0.154

    
    @staticmethod
    def H_SO():
        """
        Spin-orbit coupling Hamiltonian for SnV center.

        Returns:
            Qobj: Spin-orbit coupling Hamiltonian.
        """
        return SingleSnVHamiltonian.lambda_SO / 2 * tensor(SingleSnVHamiltonian.sigma_y, SingleSnVHamiltonian.sigma_z)

    @staticmethod
    def H_Z(B):
        """
        Zeeman splitting Hamiltonian for SnV center.

        Parameters:
            B (np.ndarray): Magnetic field vector [Bx, By, Bz].

        Returns:
            Qobj: Zeeman splitting Hamiltonian.
        """
        Bx, By, Bz = B
        H_zeeman = SingleSnVHamiltonian.gamma_e * (Bx * sigmax() + By * sigmay() + (1+2*SingleSnVHamiltonian.delta_g)*Bz * sigmaz())
        return tensor(SingleSnVHamiltonian.I,H_zeeman)
    @staticmethod
    def H_L(B):
        Bx, By, Bz = B
        H_L = -0.5*SingleSnVHamiltonian.f_g*SingleSnVHamiltonian.gamma_e * (Bz * sigmay())
        return tensor(H_L, SingleSnVHamiltonian.I)
    @staticmethod
    def H_JT(Upsilon_x, Upsilon_y):
        """
        Jahn-Teller effect Hamiltonian for SnV center.

        Parameters:
            Upsilon_x, Upsilon_y (float): Coupling constants for Jahn-Teller effect.

        Returns:
            Qobj: Jahn-Teller effect Hamiltonian.
        """
        H_JT = Qobj([[Upsilon_x, Upsilon_y], [Upsilon_y, -Upsilon_x]])
        return tensor(H_JT, SingleSnVHamiltonian.I)
    
    @staticmethod
    def H_SO_T(alpha):
        """
        Spin-orbit coupling Hamiltonian for SnV center with temperature dependency.

        Returns:
            Qobj: Spin-orbit coupling Hamiltonian.
        """
        # print(alpha[1])
        # X = alpha[0] * alpha[1]**3
        return (SingleSnVHamiltonian.lambda_SO - alpha) / 2 * tensor(SingleSnVHamiltonian.sigma_y, SingleSnVHamiltonian.sigma_z)

    @staticmethod
    def H_total(B):
        """
        Total Hamiltonian for SnV center.

        Parameters:
            B (np.ndarray): Magnetic field vector [Bx, By, Bz].
            Upsilon_x, Upsilon_y (float): Coupling constants for Jahn-Teller effect.

        Returns:
            Qobj: Total Hamiltonian.
        """
        return SingleSnVHamiltonian.H_SO() + SingleSnVHamiltonian.H_Z(B) + SingleSnVHamiltonian.H_JT(SingleSnVHamiltonian.Upsilon_x,SingleSnVHamiltonian.Upsilon_y) + SingleSnVHamiltonian.H_L(B)

    def H_total_T(B,alpha):
        """
        Total Hamiltonian for SnV center.

        Parameters:
            B (np.ndarray): Magnetic field vector [Bx, By, Bz].
            Upsilon_x, Upsilon_y (float): Coupling constants for Jahn-Teller effect.

        Returns:
            Qobj: Total Hamiltonian.
        """
        return SingleSnVHamiltonian.H_SO_T(alpha) + SingleSnVHamiltonian.H_Z(B) + SingleSnVHamiltonian.H_JT(SingleSnVHamiltonian.Upsilon_x,SingleSnVHamiltonian.Upsilon_y) + SingleSnVHamiltonian.H_L(B)
    
    def SNV_eigenEnergiesStates(B):
        H_total = SingleSnVHamiltonian.H_total(B)
        E_I, vec_I = H_total.eigenstates()

        return E_I, vec_I
    
    def SNV_eigenEnergiesStates_T(B,alpha):
        H_total = SingleSnVHamiltonian.H_total_T(B,alpha)
        E_I, vec_I = H_total.eigenstates()

        return E_I, vec_I

    @staticmethod
    def simpleMWint(Bmw):
        """
        Compute interaction Hamiltonian, with MW vector Bmw defined in NV center frame,
        including detuning from the resonance frequency.

        Parameters
        ----------
        Bmw : numpy.ndarray
            Magnetic field vector.

        Returns
        -------
        Qobj
            Interaction Hamiltonian.
        """
        # Magnetic field coupling terms
        Bx, By, Bz = Bmw
        H_zeeman = SingleSnVHamiltonian.gamma_e * (Bx * sigmax() + By * sigmay() + Bz * sigmaz())
        return tensor(SingleSnVHamiltonian.I,H_zeeman)
        return HintEl

