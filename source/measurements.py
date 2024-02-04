#!/usr/bin/python3
#* -*- coding: utf-8 -*- 

"""
@author:https://github.com/fsfetcu

Inspiratation from https://github.com//chris-galland/qutip-odmr
"""


import numpy as np
from hamiltonian import singleNVhamiltonian as snvh
from hamiltonian import SingleSnVHamiltonian as ssnvh
from vectors import vec as vec
from utils import noise, math_functions , operators
from qutip import Qobj, tensor, qeye, mesolve, basis, expect, ket2dm
from qutip import Options
import scipy.linalg as la
from qutip.qip.operations import snot
from qutip.qip.device import Processor
from qutip.operators import sigmaz, destroy,sigmax, sigmay,sigmam, sigmap, create
from qutip.qip.operations import snot
from qutip.states import basis

from alive_progress import alive_it # Not needed, just visual
from alive_progress import alive_bar
class cwODMR:
    
    """
    A module for the cw-ODMR measurements for NV and SnV.
    """

    DIMENSION = 9 # Amount of states in the NV center
    PL_BASELINE = 1.5e6  # Adjust this value as needed
    PL_DROP = 0.1 * PL_BASELINE  # 10% drop in PL count

    
    @staticmethod
    def ODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth):
        """
        Returns ESR transition strengths for a single NV center.

        Parameters
        ----------
        MWfreq : numpy.ndarray
            Array of frequencies.
        MWvec : qt.Qobj
            Vector of the microwave field defined in NV frame.
        Bvec : qt.Qobj
            Vector of the magnetic field defined in NV frame.
        Evec : qt.Qobj
            Vector of the electric field defined in NV frame.
        Linewidth : float   
            Linewidth of the transition (depends on experimental setup).
        
        Returns
        -------
        numpy.ndarray
            Transition strengths.
        """

        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)  # Eigenenergies and eigenvectors
        Hint = snvh.GShamiltonianMWint(MWvec)  # Interaction Hamiltonian

        # Calculate transition strengths
        for i in range(cwODMR.DIMENSION):  # Sweep over all initial states
            freq_i = eigenenergies[i]  
            psi_i = eigenstates[i]

            for j in range(cwODMR.DIMENSION):  
                freq_j = eigenenergies[j]  
                psi_j = eigenstates[j]  

                # Transition matrix element and transition amplitude calculation <j|Hint|i>
                TME = (psi_j.dag() * Hint * psi_i).data.toarray()[0,0]
                TA = np.abs(TME)**2

                TS = TA * math_functions.lorentzian(MWfreq, np.abs(freq_j - freq_i), Linewidth)
                
                Tstrength += TS

        return Tstrength

    def noisyODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth):
        return cwODMR.ODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth) + noise.NoiseOf(cwODMR.ODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth), 'gaussian', 0.05)

    @staticmethod
    def nvODMR(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
        """
        Returns ESR transition strengths for a NV ensemble.
        """
        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        # Convert to cartesian coordinates
        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

         
        for MWvec, Bvec, Evec in alive_it(zip(MWvector_list, Bvector_list, Evector_list)):
            Tstrength += cwODMR.ODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth)

        n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
        return Tstrength / n_NV

    @staticmethod
    def noisy_nvODMR(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
        """
        Similar to nvODMR, but with added noise. See help(cwODMR.nvODMR) and help(utils.addNoiseTo) for details.
        """
        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        # Convert to cartesian coordinates
        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

        for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
            Tstrength = Tstrength + cwODMR.ODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth) \
                        + noise.NoiseOf(Tstrength, 'gaussian', 0.02)

        n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
        return Tstrength / n_NV
   
    @staticmethod
    def ODMRsingleNV_lock_in(MWfreq, MWvec, Bvec, Evec, Linewidth):
        """
        Returns ESR lock-in transition strengths for a single NV center.
        """

        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)  # Eigenenergies and eigenvectors
        Hint = snvh.GShamiltonianMWint(MWvec)  # Interaction Hamiltonian

        # Calculate transition strengths
        for i in range(cwODMR.DIMENSION):  # Sweep over all initial states
            freq_i = eigenenergies[i]  
            psi_i = eigenstates[i]

            for j in range(i, cwODMR.DIMENSION):  
                freq_j = eigenenergies[j]  
                psi_j = eigenstates[j]  

                # Transition matrix element and transition amplitude calculation <j|Hint|i>
                TME = (psi_j.dag() * Hint * psi_i).data.toarray()[0,0]
                TA = np.abs(TME)**2

                TS = TA * math_functions.dispersive_lorentzian(MWfreq, np.abs(freq_j - freq_i), Linewidth)
                
                Tstrength += TS

        return Tstrength

    @staticmethod
    def nvODMR_lock_in(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
        """
        Returns ESR lock-in transition strengths for a NV ensemble.
        """
        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        # Convert to cartesian coordinates
        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

        for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
            Tstrength += cwODMR.ODMRsingleNV_lock_in(MWfreq, MWvec, Bvec, Evec, Linewidth)

        n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
        return Tstrength / n_NV

    @staticmethod
    def noisy_nvODMR_lock_in(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
        """
        Returns ESR lock-in transition strengths for a NV ensemble with noise.
        """
        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        # Convert to cartesian coordinates
        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

        for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
            print(MWvec)
            Tstrength = Tstrength + cwODMR.ODMRsingleNV_lock_in(MWfreq, MWvec, Bvec, Evec, Linewidth) \
                                 + noise.NoiseOf(Tstrength, 'gaussian', 0.02)

        n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
        return Tstrength / n_NV


class pulsedODMR:

    """
    A module for the pulsed-ODMR measurements for NV..
    """
    @staticmethod
    def pulsedODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth):
        """
        Calculates the pulsed ESR transition strengths for a single NV center using pulsed ESR.
        This is equivalent to a Pi pulse on the transition of interest (2.87 GHz, |g, ms=0> to |g, ms= +-1>)

        Parameters
        ----------
        MWfreq : numpy.ndarray
            Array of frequencies.
        MWvec : qt.Qobj
            Vector of the microwave field defined in NV frame.
        Bvec : qt.Qobj
            Vector of the magnetic field defined in NV frame.
        Evec : qt.Qobj
            Vector of the electric field defined in NV frame.
        Linewidth : float
            Linewidth of the transition (depends on experimental setup).

        Returns
        -------
        numpy.ndarray
            Transition strengths.
        """

        
        nMW = len(MWfreq)  # Number of microwave frequency points
        Tstrength = np.zeros(nMW)  # Transition strength

        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)
        H_free = snvh.free_hamiltonian(Bvec, Evec)
        Hint = snvh.GShamiltonianMWint(MWvec)
        
        #pi pulse duration
        temp_duration = np.pi / ((eigenenergies[3:] - eigenenergies[:3, None]))
    
        # Take the average π pulse duration across all transitions
        pi_pulse_duration = np.mean(temp_duration)
        
        # Define the time-dependent coefficient function for the pulse
        def pulse_coeff(t, args):
            """
            square pulse
            """
            return 1 if 0 <= t <= pi_pulse_duration else 0
        
        H_pulse = [Hint, pulse_coeff]
        
        
        freq_list = [eigenenergies[0] - eigenenergies[i] for i in [3,4,5,6,7,8]]
        freq_list2 = [eigenenergies[1] - eigenenergies[i] for i in [3,4,5,6,7,8]]
        freq_list3 = [eigenenergies[2] - eigenenergies[i] for i in [3,4,5,6,7,8]]        

        for i in [3,4,5,6,7,8]:#range(len(eigenstates)):  # transition states
            freq_i = eigenenergies[i]
            psi_i = eigenstates[i]
            # Calculate transition strengths
            for j in [0,1,2]: # initial states 
                freq_j = eigenenergies[j]  
                psi_j = eigenstates[j]  
                
                if np.abs(freq_j - freq_i) in np.abs(freq_list) or np.abs(freq_j - freq_i) in np.abs(freq_list2) or np.abs(freq_j - freq_i) in np.abs(freq_list3):
                    resulti = mesolve(H_pulse, psi_i, np.linspace(0, pi_pulse_duration, 100), [], [])
                    

                    # print(freq_i-freq_j,"applied")
                    psi_i = resulti.states[-1]
                    print(psi_i)
                # Matrix element of the transition <j|Hint|i> = <j|psi_end>            
                TME = (psi_j.dag()  * psi_i).data.toarray()[0,0]
                TA = np.abs(TME)**2
                TS = TA * math_functions.lorentzian(MWfreq, np.abs(freq_j - freq_i), Linewidth)
                Tstrength += TS
                
        return Tstrength
        

    def NoisypulsedODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth):
        """
        Returns pulsed ESR transition strengths for a single NV center with noise.
        """
        return pulsedODMR.pulsedODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth) + noise.NoiseOf(pulsedODMR.pulsedODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth), 'gaussian', 0.02)
    @staticmethod
    def pulsednvODMR(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
        """
        Calculates pulsed ESR transition strengths for an NV ensemble.
        """
        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)
        
        for MWvec, Bvec, Evec in alive_it(zip(MWvector_list, Bvector_list, Evector_list)):
            
            Tstrength += pulsedODMR.pulsedODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth)

        n_NV = len(Bvector_list)  # Number of NV orientations in ensemble
        return Tstrength / n_NV

    @staticmethod
    def noisy_pulsednvODMR(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):
        """
        Calculates pulsed ESR transition strengths for an NV ensemble with noise.
        """
        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

        for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
            Tstrength = Tstrength + pulsedODMR.pulsedODMRsingleNV(MWfreq, MWvec, Bvec, Evec, Linewidth) \
                                 + noise.NoiseOf(Tstrength, 'gaussian', 0.02)

        n_NV = len(Bvector_list)
        return Tstrength / n_NV

    # @staticmethod
    # def ram(Linewidth,Bvec,Evec):
        

    #     T2 = 1 / (np.pi * Linewidth)  # T2 is in microseconds
    #     # Eigenenergies and eigenvectors
    #     #simple hamiltonian
    #     H_free = snvh.simpleFree(Bvec, Evec)
        
    #     eigenenergies, eigenstates = snvh.simpleHamiltonEigen(Bvec, Evec)

    #     def choose_transition(a,b):
    #         """
    #         Qubit frequency between two states. 
    #         """
    #         resonant = np.abs(eigenenergies[a] - eigenenergies[b])
    #         return resonant
            
    #     resonant = choose_transition(1,0)
    #     a = destroy(2)
    #     print(a)
    #     Hadamard = snot()
    #     states = Qobj(H_free[1:3, 1:3]).eigenstates()
    #     plus_state = (basis(2,1) - basis(2,0)).unit()
    #     tlist = np.linspace(0.00, 0.2,1000)
    #     # T2 = 5
    #     processor = Processor(1, t2=T2)
    #     processor.add_control(sigmaz()*resonant/2)
    #     processor.pulses[0].coeff = np.ones(len(tlist))
    #     processor.pulses[0].tlist = tlist
    #     result = processor.run_state(
    #         plus_state,analytical = False,noisy = True, solver = 'mesolve', e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard])

    #     return tlist , result.expect, T2


    @staticmethod
    def ram2(tau,MWvec,Bvec,Evec,Linewidth,t_points):
        T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
        def choose_transition(a,b):
            """
            Qubit frequency between two states. 
            """
            resonant = np.abs(eigenenergies[a] - eigenenergies[b])
            return resonant
        # Eigenenergies and eigenvectors
        #simple hamiltonian
        H_free = snvh.simpleFree(Bvec, Evec)
        eigenenergies, eigenstates = snvh.simpleHamiltonEigen(Bvec, Evec)
        resonant = choose_transition(1,0)
        # states = Qobj(H_free[1:3, 1:3]).eigenstates()
        H_free = sigmaz()*resonant/2
        # Define the pi/2 pulse duration

        
        Hadamard = snot()
        a = destroy(2)
        e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard]
        c_ops = [np.sqrt(1/(T2_star))*sigmaz(),sigmam() * np.sqrt(1/20)]
        superposition_state = (basis(2,0) - basis(2,1)).unit()

        # Evolve freely for time tau
        for t in tau:
            result = mesolve(H_free, superposition_state, np.linspace(0, t, t_points), c_ops = c_ops, e_ops= e_ops)

            # state = result.states[-1]
            # print(state)

            # result = mesolve(H_free, superposition_state, np.linspace(t, 2*t, 100), c_ops = c_ops, e_ops= e_ops)
            
            # state = sigmax() * state_after_echo
            # result = mesolve(H_free, superposition_state, np.linspace(0, t, 50), c_ops = [np.sqrt(1/T2_star)*sigmax()], e_ops= e_ops)
        return tau,result.expect,T2_star
                


    def hahn_echo(tau, MWvec, Bvec,Evec, Linewidth,t_points):
        T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
        def choose_transition(a,b):
                """
                Qubit frequency between two states. 
                """
                resonant = np.abs(eigenenergies[a] - eigenenergies[b])
                return resonant
        # Eigenenergies and eigenvectors
        #simple hamiltonian
        H_free = snvh.simpleFree(Bvec, Evec)
        eigenenergies, eigenstates = snvh.simpleHamiltonEigen(Bvec, Evec)
        
        resonant = choose_transition(1,0)
        H_free = sigmaz()*resonant/2
        # Define the pi/2 pulse duration
        pi_pulse_duration = np.pi / (resonant)
        Hint = ssnvh.simpleMWint(MWvec)
        
        Hadamard = snot()
        a = destroy(2)
        e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard,sigmap(),sigmam()]
        c_ops = [np.sqrt(1/(T2_star))*sigmaz(),sigmam() * np.sqrt(1/(10*T2_star))]
        superposition_state = (basis(2,0) - basis(2,1)).unit()

        # Evolve freely for time tau
        # result = mesolve(H_free, superposition_state, np.linspace(0, tau/2, 100), c_ops = [np.sqrt(1/(T2_star))*sigmaz()], e_ops= [])
        
        # result3 = mesolve(H_free, result.states[-1], np.linspace(0, tau/2 , 100), c_ops = c_ops, e_ops= e_ops)
        for t in tau:
            result1 = mesolve(H_free, superposition_state, np.linspace(0, t/2, t_points), c_ops = c_ops, e_ops= e_ops)
            new_state = (result1.expect[3][-1]*basis(2,0) + result1.expect[2][-1]*basis(2,1)).unit()
            
            result1 = mesolve(H_free, new_state, np.linspace(0, t/2, t_points), c_ops = c_ops, e_ops= e_ops)

        return tau,result1.expect, T2_star       

    def relaxation(tau_range,MWvec,Bvec,Evec,Linewidth,t_points):
        T1 = 1/(np.pi * Linewidth)  # T1 is in microseconds
        resonant = 2870
        H_free = sigmaz()*resonant/2
        # Define the pi/2 pulse duration

        
        Hadamard = snot()
        a = create(2)
        e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard]
        c_ops = [sigmam() * np.sqrt(1/(20*T1))]
        superposition_state = (basis(2,0)).unit()

        # Evolve freely for time tau
        # result = mesolve(H_free, superposition_state, np.linspace(0, tau/2, 100), c_ops = [np.sqrt(1/(T2_star))*sigmaz()], e_ops= [])
        
        # result3 = mesolve(H_free, result.states[-1], np.linspace(0, tau/2 , 100), c_ops = c_ops, e_ops= e_ops)
        for tau in tau_range:
            result3 = mesolve(H_free, superposition_state, np.linspace(0, tau, t_points), c_ops = c_ops, e_ops= e_ops)
        return tau_range,result3.expect,T1

    @staticmethod
    def ramsey_secq(tau_range, thetaMW, phiMW, B0, thetaB, phiB, E0, thetaE, phiE, Linewidth):

        nMW = len(tau_range)  
        fluorescence = np.zeros(nMW)  

        Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
        Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
        MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

        for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
            fluorescence = fluorescence + pulsedODMR.ramseySingleNV(tau_range, MWvec, Bvec, Evec, Linewidth)

        n_NV = len(Bvector_list)
        return fluorescence / n_NV

    



class SnV_ODMR():
    
    def singleSnVodmr(MWfreq, MWvec, Bvec, Linewidth):
        """
        Returns ESR transition strengths for a single SnV center.

        Parameters
        ----------
        MWfreq : numpy.ndarray
            Array of frequencies.
        MWvec : qt.Qobj
            Vector of the microwave field defined in SnV frame.
        Bvec : qt.Qobj
            Vector of the magnetic field defined in SnV frame.
        Linewidth : float   
            Linewidth of the transition (depends on experimental setup).
        
        Returns
        -------
        numpy.ndarray
            Transition strengths.
        """

        nMW = len(MWfreq)  
        Tstrength = np.zeros(nMW)  

        eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)  # Eigenenergies and eigenvectors
        Hint = ssnvh.simpleMWint(MWvec)  # Interaction Hamiltonian
        print(eigenenergies)
        # Calculate transition strengths
        for i in [0,1,2,3]:  # Sweep over all initial states
            freq_i = eigenenergies[i]  
            psi_i = eigenstates[i]

            for j in range(4):  
                freq_j = eigenenergies[j]  
                psi_j = eigenstates[j]  

                if i!= j:
                    # Transition matrix element and transition amplitude calculation <j|Hint|i>
                    TME = (psi_j.dag() *Hint* psi_i).data.toarray()[0,0]
                    TA = np.abs(TME)**2

                    TS = TA * math_functions.lorentzian(MWfreq, (freq_j - freq_i), Linewidth)
                    
                    Tstrength += TS

        return Tstrength

    @staticmethod
    def NoisypulsedODMRsingleNV(MWfreq, MWvec, Bvec, Linewidth):
        return SnV_ODMR.singleSnVodmr(MWfreq, MWvec, Bvec, Linewidth) + noise.NoiseOf(SnV_ODMR.singleSnVodmr(MWfreq, MWvec, Bvec, Linewidth), 'gaussian', 0.02)
    
    def snv_pulsed2(MWfreq, MWvec, Bvec, Linewidth):
        nMW = len(MWfreq)  # Number of microwave frequency points
        Tstrength = np.zeros(nMW)  # Transition strength array

        # Calculate eigenenergies and eigenstates
        eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)

        # Interaction Hamiltonian
        Hint = ssnvh.simpleMWint(MWvec)

        # Calculate π pulse duration based on the transition between the first two eigenstates
        pi_pulse_duration = np.pi / (eigenenergies[1] - eigenenergies[0])

        # Define the time-dependent coefficient function for the pulse
        def pulse_coeff(t, args):
            return 1 if 0 <= t <= pi_pulse_duration else 0

        # Define the Hamiltonian for the pulse
        H_pulse = [Hint, pulse_coeff]

        for i in range(4):  # Sweep over all states
            for j in range(4):
                if i != j:
                    # Transition frequency
                    transition_freq = np.abs(eigenenergies[j] - eigenenergies[i])

                    # Find indices where MW frequency is resonant with the transition within the linewidth
                    resonant_indices = np.where(np.abs(MWfreq - transition_freq) <= Linewidth)[0]

                    # Apply π-pulse only if there is resonance
                    if len(resonant_indices) > 0:
                        # Simulate the system under the pulse
                        resulti = mesolve(H_pulse, eigenstates[i], np.linspace(0, np.pi/np.abs((transition_freq)), 300), [], [])
                        psi_i = resulti.states[-1]

                        # Calculate the transition matrix element at resonant frequencies
                        TME = (eigenstates[j].dag() * psi_i).data.toarray()[0,0]
                        TA = np.abs(TME)**2

                        # Update the transition strength only at the resonant frequencies
                        for idx in resonant_indices:
                            Tstrength[idx] += TA * math_functions.lorentzian(MWfreq[idx], transition_freq, Linewidth)

        return Tstrength
    @staticmethod
    def snv_pulsed(MWfreq, MWvec, Bvec, Linewidth):
                
        nMW = len(MWfreq)  # Number of microwave frequency points
        Tstrength = np.zeros(nMW)  # Transition strength
        SX = tensor(ssnvh.I,ssnvh.sigma_x)
        # Eigenenergies and eigenvectors

        H_free = ssnvh.H_total(Bvec)
        eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)  # Eigenenergies and eigenvectors
        print(eigenenergies)
        Hint = ssnvh.simpleMWint(MWvec)  # Interaction Hamiltonian
        #pi pulse duration
        temp_duration = np.pi / ((eigenenergies[1]-eigenenergies[0]))
    
        # Take the average π pulse duration across all transitions
        pi_pulse_duration = np.mean(temp_duration)
        
        # Define the time-dependent coefficient function for the pulse
        def pulse_coeff(t, args):
            """
            square pulse
            """
            return 1 if 0 <= t <= pi_pulse_duration else 0
        
        H_pulse = [Hint, pulse_coeff]
        print(eigenenergies[1]-eigenenergies[0])
        
        for i in [0,1,2,3]:  # Sweep over all initial states
            freq_i = eigenenergies[i]
            psi_i = eigenstates[i]
            # Calculate transition strengths
            for j in [0,1,2,3]:  
                freq_j = eigenenergies[j]  
                psi_j = eigenstates[j]  
                if i!=j:
                    
                    if np.abs(freq_j - freq_i) == np.abs(eigenenergies[1]-eigenenergies[0]):
                        print(psi_i)
                        resulti = mesolve(H_pulse, psi_j, np.linspace(0, pi_pulse_duration, 300), [], [])
                        print(resulti.states[-1])
                        # print(SX * psi_i)
                        # print(SX*psi_i == resulti.states[-1])
                        print(freq_i-freq_j,"applied")
                        # psi_j = resulti.states[-1]
                        # print(psi_i == psi_x) 
                    # Matrix element of the transition <j|Hint|i> = <j|psi_end>   
                                
                        TME = (psi_j.dag() * resulti.states[-1]).data.toarray()[0,0]
                        TA = np.abs(TME)**2
                        TS = TA *math_functions.lorentzian(MWfreq, ((freq_j - freq_i)), Linewidth)
                        Tstrength += TS
                    else: 
                        TME = (psi_j.dag()* psi_i).data.toarray()[0,0]
                        TA = np.abs(TME)**2
                        TS = TA *math_functions.lorentzian(MWfreq, (freq_j - freq_i), Linewidth)
                        Tstrength += TS
                
        return Tstrength

    def ram2(tau,MWvec,Bvec,Linewidth,t_points):
            T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
            def choose_transition(a,b):
                """
                Qubit frequency between two states. 
                """
                resonant = np.abs(eigenenergies[a] - eigenenergies[b])
                return resonant
            # Eigenenergies and eigenvectors
            #simple hamiltonian
            H_free = ssnvh.H_total(Bvec)
            eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)
            
            resonant = choose_transition(1,0)
            # states = Qobj(H_free[1:3, 1:3]).eigenstates()
            print(resonant/1e9)
            H_free = sigmaz()*resonant/2
            print(H_free)
            # Define the pi/2 pulse duration
            
            Hadamard = snot()
            a = destroy(2)
            e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard]
            c_ops = [np.sqrt(1/(T2_star))*sigmaz(),sigmam() * np.sqrt(1/(10*T2_star))]#[np.sqrt(1/(T2_star))*sigmaz()]
            superposition_state = (basis(2,0) - basis(2,1)).unit() # pi/2 of |0>

            # Evolve freely for time tau
            
            for t in tau:
                result = mesolve(H_free, superposition_state, np.linspace(0, t, t_points), c_ops = c_ops, e_ops= e_ops)

                # state = result.states[-1]
                # print(state)

                # result = mesolve(H_free, superposition_state, np.linspace(t, 2*t, 100), c_ops = c_ops, e_ops= e_ops)
                
                # state = sigmax() * state_after_echo
                # result = mesolve(H_free, superposition_state, np.linspace(0, t, 50), c_ops = [np.sqrt(1/T2_star)*sigmax()], e_ops= e_ops)
            return tau,result.expect,T2_star

    def hahn_echo(tau, MWvec, Bvec, Linewidth,t_points):
        T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
        def choose_transition(a,b):
                """
                Qubit frequency between two states. 
                """
                resonant = np.abs(eigenenergies[a] - eigenenergies[b])
                return resonant
        # Eigenenergies and eigenvectors
        #simple hamiltonian
        H_free = ssnvh.H_total(Bvec)
        eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)
        
        resonant = choose_transition(1,0)
        print(resonant)
        H_free = sigmaz()*resonant/2
        # Define the pi/2 pulse duration
        pi_pulse_duration = np.pi / (resonant)
        Hint = ssnvh.simpleMWint(MWvec)
        
        Hadamard = snot()
        a = destroy(2)
        e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard,sigmap(),sigmam()]
        c_ops = [np.sqrt(1/(T2_star))*sigmaz(),sigmam() * np.sqrt(1/(10*T2_star))]
        superposition_state = (basis(2,0) - basis(2,1)).unit()

        # Evolve freely for time tau
        # result = mesolve(H_free, superposition_state, np.linspace(0, tau/2, 100), c_ops = [np.sqrt(1/(T2_star))*sigmaz()], e_ops= [])
        
        # result3 = mesolve(H_free, result.states[-1], np.linspace(0, tau/2 , 100), c_ops = c_ops, e_ops= e_ops)
        for t in tau:
            result1 = mesolve(H_free, superposition_state, np.linspace(0, t/2, t_points), c_ops = c_ops, e_ops= e_ops)
            new_state = (result1.expect[3][-1]*basis(2,0) + result1.expect[2][-1]*basis(2,1)).unit()
            
            result1 = mesolve(H_free, new_state, np.linspace(0, t/2, t_points), c_ops = c_ops, e_ops= e_ops)

        return tau,result1.expect, T2_star       

    def relaxation(tau_range,MWvec,Bvec,Linewidth,t_points):
        T1 = 1/(np.pi * Linewidth)  # T1 is in microseconds
        def choose_transition(a,b):
                """
                Qubit frequency between two states. 
                """
                resonant = np.abs(eigenenergies[a] - eigenenergies[b])
                return resonant
        eigenenergies, eigenstates = ssnvh.SNV_eigenEnergiesStates(Bvec)
        resonant = choose_transition(1,0)
        H_free = sigmaz()*resonant/2
        # Define the pi/2 pulse duration

        
        Hadamard = snot()
        a = create(2)
        e_ops=[a.dag()*a, Hadamard*a.dag()*a*Hadamard]
        c_ops = [sigmam() * np.sqrt(1/(20*T1))]
        superposition_state = (basis(2,0)).unit()

        # Evolve freely for time tau
        # result = mesolve(H_free, superposition_state, np.linspace(0, tau/2, 100), c_ops = [np.sqrt(1/(T2_star))*sigmaz()], e_ops= [])
        
        # result3 = mesolve(H_free, result.states[-1], np.linspace(0, tau/2 , 100), c_ops = c_ops, e_ops= e_ops)
        for tau in tau_range:
            result3 = mesolve(H_free, superposition_state, np.linspace(0, tau, t_points), c_ops = c_ops, e_ops= e_ops)
        return tau_range,result3.expect,T1