@staticmethod
    def ramseySingleNV3(tau_range, MWvec, Bvec, Evec, Linewidth):
        T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
        nMW = len(tau_range)  # Number of microwave frequency points

        # Eigenenergies and eigenvectors
        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)
        Hint = snvh.GShamiltonianMWint(MWvec)
        H_free = snvh.free_hamiltonian(Bvec, Evec)

        # Define the pi/2 pulse duration
        pi2_pulse_duration = np.pi / (2 * ((eigenenergies[3:] - eigenenergies[:3, None])))
        pi2_pulse_duration = np.mean(pi2_pulse_duration)

        # Define the time-dependent function for the concatenated pulse train
        def pulse_train(t, args):
            if 0 <= t <= pi2_pulse_duration or (pi2_pulse_duration + args['tau']) <= t <= (2 * pi2_pulse_duration + args['tau']):
                return 1  # During π/2 pulses
            else:
                return 0  # During free evolution

        # Prepare the total Hamiltonian for the simulation
        H_total = [H_free, [Hint, pulse_train]]

        transition_probabilities = np.zeros(len(tau_range))


        # Loop over tau_range and perform the simulation
        for idx, tau in enumerate(tau_range):
            # Set the additional parameter 'tau' for the current free evolution time
            args = {'tau': tau}

            # Total time for the simulation (2 pulses + free evolution)
            total_time = 2 * pi2_pulse_duration + tau
            timesteps = np.linspace(0, total_time, 1000)

            # Perform the simulation
            initial_state = eigenstates[0]  # Set your initial state accordingly
            result = mesolve(H_total, initial_state, timesteps, [], [],args=args)

            final_state = result.states[-1]
            for i in [3,4,5]:
                TME = (eigenstates[i].dag() * final_state).data.toarray()[0,0]
                TA = np.abs(TME) ** 2
                transition_probabilities[idx] += TA
        return transition_probabilities 



    @staticmethod
    def ramseySingleNV2(tau_range, MWvec, Bvec, Evec, Linewidth):
        T2_star = 1 / (np.pi * Linewidth)  # T2 is in microseconds
        nMW = len(tau_range)  # Number of microwave frequency points
        Tstrength = np.zeros(nMW)  # Transition strength

        # Eigenenergies and eigenvectors
        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)
        Hint = snvh.GShamiltonianMWint(MWvec)
        H_free = snvh.free_hamiltonian(Bvec, Evec)

        # Define the pi/2 pulse duration
        pi2_pulse_duration = np.pi / (2 * ((eigenenergies[3:] - eigenenergies[:3, None])))
        pi2_pulse_duration = np.mean(pi2_pulse_duration)

        # Define the time-dependent coefficient function for the pulse
        def pulse_coeff(t, args):
            """Square pulse"""
            return 1 if 0 <= t <= pi2_pulse_duration else 0

        H_pulse = [Hint+H_free, pulse_coeff]

        freq_list = [eigenenergies[0] - eigenenergies[i] for i in [3,4,5,6,7,8]]
        freq_list2 = [eigenenergies[1] - eigenenergies[i] for i in [3,4,5,6,7,8]]
        freq_list3 = [eigenenergies[2] - eigenenergies[i] for i in [3,4,5,6,7,8]]        

        # Collect transition frequencies for resonance check
        transition_frequencies = eigenenergies[3:] - eigenenergies[:3, None]

        print(transition_frequencies)
        for idx, tau in alive_it(enumerate(tau_range)):
            for i in [0,1,2]:  # Only the ground states
                psi_i = eigenstates[i]
                
                # Apply the first pi/2 pulse, free evolution, and second pi/2 pulse if in resonance
                for j in [3,4,5,6,7,8]:  # Only the excited states
                    freq_ij = eigenenergies[j] - eigenenergies[i]
                    if i != j:
                        # Check if transition is within resonance using a tolerance
                            resulti = mesolve(H_pulse, psi_i, np.linspace(0, pi2_pulse_duration, 50), [], [])
                            psi_i = resulti.states[-1]
                            
                            # Free evolution until tau
                            result_free = mesolve(H_free, psi_i, np.linspace(0, tau, 50), [], [])
                            psi_i = result_free.states[-1]
                            
                            # Apply second pi/2 pulse
                            resulti = mesolve(H_pulse, psi_i, np.linspace(0, pi2_pulse_duration, 50), [], [])
                            psi_i = resulti.states[-1]
                        
                            # Calculate transition strength and add to the total strength
                    TME = (eigenstates[j].dag() * psi_i).data.toarray()[0,0]
                    TA = np.abs(TME) ** 2
                    
                    B_magnitude = np.linalg.norm(Bvec)  # Calculate the magnitude of the magnetic field
                    phi = freq_ij * tau
                    Tstrength[idx] += TA * math_functions.decay_envelope(tau, freq_ij, T2_star,phi=2*pi2_pulse_duration)
                                                 
        return Tstrength
    
    def ramsey_spectroscopy(tau_range, MWvec, Bvec, Evec, Linewidth):
        def apply_pi2_pulse(psi):
            Sx = snvh.S_x
            R = (-1j * np.pi/2 * Sx).expm()
            return R * psi

        def evolve_state(psi, H, tau):
            times = np.linspace(0, tau, 300)
            result = mesolve(H, psi, times, [], [])
            return result.states[-1]

        def decay_envelope(tau, delta, T2_star, phi=0):
            return np.exp(-tau/T2_star - 1j * 2 * np.pi * delta * tau + 1j * phi)

        def ramsey_fringes(tau, delta, T2_star):
            return np.exp(-tau/T2_star) * np.cos(2 * np.pi * delta * tau)
        
        T2_star = 1 / (np.pi * Linewidth)
        nMW = len(tau_range)
        Tstrength = np.zeros(nMW)
        eigenenergies, eigenstates = snvh.simpleHamiltonEigen(Bvec, Evec)
        Hint = snvh.simpleMWint(MWvec)
        H_free = snvh.simpleFree(Bvec, Evec)
        rabi = 2870
        pi2_pulse_duration = np.pi / (2 * rabi)

        for idx, tau in enumerate(tau_range):
            psi = eigenstates[0]
            psi = apply_pi2_pulse(psi)
            psi = evolve_state(psi, H_free, tau)
            psi = apply_pi2_pulse(psi)

            for j in range(len(eigenstates)):
                freq_j = eigenenergies[j]
                psi_j = eigenstates[j]
                TME = (psi_j.dag() * psi).data.toarray()[0,0]
                TA = np.abs(TME) ** 2
                Tstrength[idx] += TA * math_functions.decay_envelope(tau, np.abs(freq_j), T2_star,phi=tau)

        return Tstrength



    @staticmethod
    def ramseySingleNV(tau_range, MWvec, Bvec, Evec, Linewidth):
        nMW = len(tau_range)  # Number of microwave frequency points
        Tstrength = np.zeros(nMW)  # Transition strength
        # Eigenenergies and eigenvectors

        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)
        Hint = snvh.GShamiltonianMWint(MWvec)
        H_free = snvh.free_hamiltonian(Bvec, Evec)
        #pi pulse duration
        temp_duration = 0.5*np.pi / ((eigenenergies[3:] - eigenenergies[:3, None]))

        # Take the average π pulse duration across all transitions
        pi2_pulse_duration = np.mean(temp_duration)
        T2_star = 1/(np.pi*Linewidth) # T2 is in microseconds
        # Define the time-dependent coefficient function for the pulse
        def pulse_coeff(t, args):
            """
            square pulse
            """
            return 1 if 0 <= t <= pi2_pulse_duration else 0

        H_pulse = [Hint+H_free, pulse_coeff]
        
        for idx, tau in alive_it(enumerate(tau_range)):

            psi_i = (eigenstates[2])

            freq_i = (eigenenergies[2])
            

            #apply first pi/2 pulse
            resulti = mesolve(H_pulse, psi_i, np.linspace(0, pi2_pulse_duration, 500), [], [])     
            psi_i = resulti.states[-1]

            #then free evolution untill tau
            result_free = mesolve(H_free, psi_i, np.linspace(0, tau, 500), [], [])
            psi_i = result_free.states[-1]

            #then apply second pi/2 pulse
            resulti = mesolve(H_pulse, psi_i, np.linspace(0, pi2_pulse_duration, 500), [], [])
            psi_i = resulti.states[-1]

            # Calculate transition strengths

            for j in range(len(eigenstates)):
                freq_j = eigenenergies[j]
                psi_j = eigenstates[j]

                # Matrix element of the transition <j|Hint|i> = <j|psi_end>
                TME = (psi_j.dag() * psi_i).data.toarray()[0,0]
                TA = np.abs(TME)**2

                TS = TA * math_functions.decay_envelope(tau, np.abs(freq_j), T2_star*10,phi=0)
                #phi = 2*pi * gammaNV * B (length of B) * tau
                # phi = np.sqrt(np.dot(np.array(Bvec),np.array(Bvec))) * 2*np.pi * snvh.gammaNV * tau
                Tstrength[idx] += TS 
                
        return Tstrength




    @staticmethod
    def ramseySingleNV11(tau_range, MWvec, Bvec, Evec, Linewidth):
        # pi/2 pulse duration for the Rabi frequency of 2.87 GHz
        pi_over_2_pulse_duration = (np.pi / 2) / (2.87e9 * 2 * np.pi)

        # Define the time-dependent coefficient function for the pulse
        def pulse_coeff(t, args):
            return 1 if 0 <= t <= pi_over_2_pulse_duration else 0
        
        # Assuming GShamiltonianMWint and free_hamiltonian methods are defined in your class
        H_pulse = [snvh.GShamiltonianMWint(MWvec), pulse_coeff]
        H_free = snvh.free_hamiltonian(Bvec, Evec)

        eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)  # Eigenenergies and eigenvectors
        ms0_eigenenergies = [0, 1, 2]

        # Create a mixed state of the ms=0 hyperfine states
        psi0 = eigenstates[ms0_eigenenergies[0]]

        fluorescence_vs_tau = []
        for tau in tau_range:
            # Apply first MW π/2 pulse
            tlist_pulse = np.linspace(0, pi_over_2_pulse_duration, 100)
            result_after_pulse = mesolve(H_pulse, psi0, tlist_pulse, [], [])
            psi_after_pulse1 = result_after_pulse.states[-1]

            # Evolve freely under H_free for time tau
            tlist_free = np.linspace(0, tau, 100)
            result_free_evolution = mesolve(H_free, psi_after_pulse1, tlist_free, [], [])
            psi_after_free_evolution = result_free_evolution.states[-1]

            # Apply second MW π/2 pulse
            result_after_pulse = mesolve(H_pulse, psi_after_free_evolution, tlist_pulse, [], [])
            psi_after_pulse2 = result_after_pulse.states[-1]

            # Calculate populations of each state
            final_state = psi_after_pulse2
            state_populations = [np.abs((eigenstates[j].dag() * final_state).norm())**2 for j in ms0_eigenenergies]
            # Calculate fluorescence signal
            fluorescence_signal = sum(state_populations[i] for i in ms0_eigenenergies)
            fluorescence_vs_tau.append(fluorescence_signal)

        return fluorescence_vs_tau


        # for tau in tau_range:
        #     # Apply the first π/2 pulse
        #     H_pulse = [snvh.GShamiltonianMWint(MWvec), pulse_coeff]
        #     psi0 = tensor(basis(3, 0), basis(3, 0))  # Replace with the correct initial state
        #     result = mesolve(H_pulse, psi0, np.linspace(0, pi_over_2_pulse_duration, 100), [], [])
        #     psi_after_first_pulse = result.states[-1]

        #     # Free evolution for time τ
        #     H_free = snvh.free_hamiltonian(Bvec, Evec)
        #     result_free = mesolve(H_free, psi_after_first_pulse, np.linspace(0, tau, 100), [], [])
        #     psi_after_free_evolution = result_free.states[-1]

        #     # Apply the second π/2 pulse
        #     result = mesolve(H_pulse, psi_after_free_evolution, np.linspace(0, pi_over_2_pulse_duration, 100), [], [])
        #     psi_end = result.states[-1]

        #     # Calculate fluorescence
        #     # Here you would calculate the expected fluorescence signal based on psi_end
        #     # This would involve determining the probability of the NV center being in the m_s = 0 state
        #     # Assuming fluorescence signal is proportional to this probability
        #     fluorescence_signal = np.abs(psi_end.overlap(psi0))**2
        #     fluorescence_vs_tau.append(fluorescence_signal)

        # return np.array(fluorescence_vs_tau)

    # @staticmethod
    # def ramseySingleNV(tau_range, MWvec, Bvec, Evec, Linewidth):
    #     """
    #     Run a Ramsey sequence for varying evolution times (tau_range) and calculate the fluorescence.
    #     """
    #     # Constants for NV center
    #     gamma_nv = 2 * np.pi * 2.87e9  # Gyromagnetic ratio for NV center in rad/s

    #     # Pi/2 pulse duration
    #     pi_over_2_pulse_duration = (np.pi/2) / gamma_nv

    #     # Initialize fluorescence signal array
    #     fluorescence_signal = np.zeros_like(tau_range)

    #     # Eigenenergies and eigenvectors of the NV center Hamiltonian
    #     eigenenergies, eigenstates = snvh.NV_eigenEnergiesStates(Bvec, Evec)

    #     # Define the time-dependent coefficient function for the pulse
    #     def pulse_coeff(t, args):
    #         return 1 if 0 <= t <= pi_over_2_pulse_duration else 0

    #     # Define the interaction Hamiltonian
    #     Hint = snvh.GShamiltonianMWint(MWvec,10e6)
    #     print(Hint)
    #     T2_star = 1 / (np.pi * Linewidth)  # Convert linewidth (FWHM) to T2* (dephasing time)
    #     # Assuming the function generate_c_opsT12 generates the correct collapse operators
    #     c_ops = operators.generate_c_opsT12(9, 9, 1e-6, T2_star)  # Collapse operators for dephasing and relaxation
    #     print(c_ops)
    #     # Define the initial state |ms=0> for the ground state manifold
    #     # Assuming the 0th index corresponds to the |ms=0> state
    #     psi0 = tensor(basis(9, 0), basis(9,0))  # Corrected initial state
    #     print(psi0)
    #     # Loop over each free evolution time
    #     for idx, tau in enumerate(tau_range):
    #         # First MW π/2 pulse to create superposition of |ms=0> and |ms=+1>
    #         H_pulse_first = [Hint, pulse_coeff]
    #         result_first_pulse = mesolve(H_pulse_first, psi0, np.linspace(0, pi_over_2_pulse_duration, 100), c_ops, [])

    #         # Free evolution under the Hamiltonian for time τ
    #         psi_after_first_pulse = result_first_pulse.states[-1]
    #         result_free_evolution = mesolve(Hint, psi_after_first_pulse, [0, tau], c_ops, [])

    #         # Second MW π/2 pulse
    #         psi_after_free_evolution = result_free_evolution.states[-1]
    #         H_pulse_second = [Hint, pulse_coeff]
    #         result_second_pulse = mesolve(H_pulse_second, psi_after_free_evolution, np.linspace(0, pi_over_2_pulse_duration, 100), c_ops, [])

    #         # Calculate fluorescence signal assuming it's proportional to the probability of being in |ms=0>
    #         psi_end = result_second_pulse.states[-1]
    #         fluorescence_signal[idx] = np.abs((psi_end.dag() * tensor(basis(9, 0), qeye(9))).norm())**2

    #     return fluorescence_signal
            

# Constants for NV center
# gamma_nv = 2 * np.pi * 2.87e9  # Gyromagnetic ratio for NV center in rad/s
# pi_over_2_pulse_duration = (np.pi/2) / gamma_nv

# # Define the detuning (e.g., +5 MHz detuning from the resonance frequency)
# detuning = 5e6  # in Hz

# # Initial state should be |ms=0>
# psi0 = tensor(basis(3, 0), qeye(3)).unit()  # This creates a density matrix

# # Define collapse operators to include relaxation and dephasing
# # Assuming Linewidth is the FWHM of the Lorentzian line shape in Hz
# T2_star = 1 / (np.pi * Linewidth)  # Convert linewidth (FWHM) to T2* (dephasing time)
# c_ops = [np.sqrt(1/T2_star) * tensor(qeye(3), qeye(3))]  # Collapse operator for dephasing

# fluorescence_vs_tau = []

# # Solver options
# opts = Options(nsteps=10000)

# for tau in tau_range:
#     # Apply the first MW π/2 pulse with detuning
#     H_pulse = singleNVhamiltonian.GShamiltonianMWint(MWvec, detuning)
#     result = mesolve(H_pulse, psi0, [0, pi_over_2_pulse_duration], c_ops, [], options=opts)
#     psi_after_first_pulse = result.states[-1]

#     # Free evolution under the Hamiltonian for time τ
#     H_free = singleNVhamiltonian.free_hamiltonian(Bvec, Evec)
#     result_free = mesolve(H_free, psi_after_first_pulse, [0, tau], c_ops, [], options=opts)
#     psi_after_free_evolution = result_free.states[-1]

#     # Apply the second MW π/2 pulse with detuning
#     result_second_pulse = mesolve(H_pulse, psi_after_free_evolution, [0, pi_over_2_pulse_duration], c_ops, [], options=opts)
#     psi_end = result_second_pulse.states[-1]

#     # Calculate fluorescence signal assuming it's proportional to the probability of being in |ms=0>
#     fluorescence_signal = np.abs(psi_end.overlap(psi0))**2
#     fluorescence_vs_tau.append(fluorescence_signal)

     # @staticmethod
    # def ramseySingleNV(tau_range, MWvec, Bvec, Evec, Linewidth):
    #     # pi/2 pulse duration for the Rabi frequency of 2.87 GHz
    #     pi_over_2_pulse_duration = (np.pi/2) / (2.87e9 * 2 * np.pi)

    #     # Define the time-dependent coefficient function for the pulse
    #     def pulse_coeff(t, args):
    #         return 1 if 0 <= t <= pi_over_2_pulse_duration else 0
        
    #     fluorescence_vs_tau = []

    #     for tau in tau_range:
    #         # Apply the first π/2 pulse
    #         H_pulse = [snvh.GShamiltonianMWint(MWvec), pulse_coeff]
    #         psi0 = tensor(basis(3, 0), basis(3, 0))  # Replace with the correct initial state
    #         result = mesolve(H_pulse, psi0, np.linspace(0, pi_over_2_pulse_duration, 100), [], [])
    #         psi_after_first_pulse = result.states[-1]

    #         # Free evolution for time τ
    #         H_free = snvh.free_hamiltonian(Bvec, Evec)
    #         result_free = mesolve(H_free, psi_after_first_pulse, np.linspace(0, tau, 100), [], [])
    #         psi_after_free_evolution = result_free.states[-1]

    #         # Apply the second π/2 pulse
    #         result = mesolve(H_pulse, psi_after_free_evolution, np.linspace(0, pi_over_2_pulse_duration, 100), [], [])
    #         psi_end = result.states[-1]

    #         # Calculate fluorescence
    #         # Here you would calculate the expected fluorescence signal based on psi_end
    #         # This would involve determining the probability of the NV center being in the m_s = 0 state
    #         # Assuming fluorescence signal is proportional to this probability
    #         fluorescence_signal = np.abs(psi_end.overlap(psi0))**2
    #         fluorescence_vs_tau.append(fluorescence_signal)

    #     return np.array(fluorescence_vs_tau)
    # # Constants for NV center
        # pi_over_2_pulse_duration = (np.pi/2) / (2.87e9 * 2 * np.pi)
        # gamma_nv = 2 * np.pi * 2.87e9  # Gyromagnetic ratio for NV center in rad/s
        # # Initial state should be |ms=0>
        # psi0 = tensor(basis(3, 0) * basis(3, 0).dag(),qeye(3))
        # print(psi0)
        # def pulse_coeff(t, args):
        #     return 1 if 0 <= t <= pi_over_2_pulse_duration else 0

        # # Define collapse operators to include relaxation and dephasing
        # # Assuming Linewidth is the FWHM of the Lorentzian line shape in Hz
        # T2_star = 1 / (np.pi * Linewidth)  # Convert linewidth (FWHM) to T2* (dephasing time)
        # c_ops = tensor(np.sqrt(1/T2_star) * qeye(3),qeye(3))  # Collapse operator for dephasing
        
        # fluorescence_vs_tau = []

        # # Solver options
        # opts = Options(nsteps=10000)

        # for tau in tau_range:
        #     # Apply the first MW π/2 pulse
        #     H_pulse = [snvh.GShamiltonianMWint(MWvec), pulse_coeff]
        #     result = mesolve(H_pulse, psi0, [0, pi_over_2_pulse_duration], c_ops, [], options=opts)
        #     psi_after_first_pulse = result.states[-1]

        #     # Free evolution under the Hamiltonian for time τ
        #     H_free = snvh.free_hamiltonian(Bvec, Evec)
        #     result_free = mesolve(H_free, psi_after_first_pulse, [0, tau], c_ops, [], options=opts)
        #     psi_after_free_evolution = result_free.states[-1]

        #     # Apply the second MW π/2 pulse
        #     H_pulse_second = [snvh.GShamiltonianMWint(MWvec), pulse_coeff]
        #     result_second_pulse = mesolve(H_pulse_second, psi_after_free_evolution, [0, pi_over_2_pulse_duration], c_ops, [], options=opts)
        #     psi_end = result_second_pulse.states[-1]

        #     # Calculate fluorescence signal assuming it's proportional to the probability of being in |ms=0>
        #     fluorescence_signal = np.abs(psi_end.overlap(psi0))**2
        #     fluorescence_vs_tau.append(fluorescence_signal)

        # return np.array(fluorescence_vs_tau)
