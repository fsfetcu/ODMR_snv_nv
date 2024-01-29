import matplotlib.pyplot as plt


import numpy as np
from hamiltonian import singleNVhamiltonian as snvh
from vectors import vec as vec
from utils import noise
from utils import math_functions 
from qutip import Qobj, tensor, qeye, mesolve, basis
from qutip import Options



class RamseySimulation:
    
    def __init__(self, pi2_pulse_duration,  initial_state, ms0_state, photon_scale_factor, free_hamiltonian,pi_pulse_duration=None, pi_pulse_hamiltonian=None, pi2_pulse_hamiltonian=None):
        """
        Initialize the Ramsey Simulation parameters.

        Parameters
        ----------
        pi2_pulse_duration : float
            Duration of the π/2 pulse.
        pi_pulse_duration : float, optional
            Duration of the π pulse, used if Hahn echo is implemented.
        initial_state : qt.Qobj
            The initial quantum state of the NV center.
        ms0_state : qt.Qobj
            The quantum state corresponding to |ms=0>.
        photon_scale_factor : float
            Scaling factor to convert probability to estimated photon count.
        free_hamiltonian : qt.Qobj
            The Hamiltonian during free evolution.
        pi_pulse_hamiltonian : qt.Qobj, optional
            The Hamiltonian for the π pulse, used if Hahn echo is implemented.
        pi2_pulse_hamiltonian : qt.Qobj, optional
            The Hamiltonian for the π/2 pulse.
        """
        self.pi2_pulse_duration = pi2_pulse_duration
        self.pi_pulse_duration = pi_pulse_duration
        self.initial_state = initial_state
        self.ms0_state = ms0_state
        self.photon_scale_factor = photon_scale_factor
        self.free_hamiltonian = free_hamiltonian
        self.pi_pulse_hamiltonian = pi_pulse_hamiltonian
        self.pi2_pulse_hamiltonian = pi2_pulse_hamiltonian

    def generate_pulse_sequence(self, evolution_times, hahn_echo=False):
        pulse_sequence = []
        for evolution_time in evolution_times:
            pulse_sequence.append({"type": "pi2", "duration": self.pi2_pulse_duration})
            if hahn_echo:
                pulse_sequence.append({"type": "free_evolution", "duration": evolution_time/2})
                pulse_sequence.append({"type": "pi", "duration": self.pi_pulse_duration})
                pulse_sequence.append({"type": "free_evolution", "duration": evolution_time/2})
            else:
                pulse_sequence.append({"type": "free_evolution", "duration": evolution_time})
            pulse_sequence.append({"type": "pi2", "duration": self.pi2_pulse_duration})
        return pulse_sequence


    def simulate_evolution(self, initial_state, hamiltonian, duration, pulse_type):
        if pulse_type in ["pi", "pi2"]:
            def pulse_coeff(t, args):
                pulse_duration = self.pi_pulse_duration if pulse_type == "pi" else self.pi2_pulse_duration
                return 1 if 0 <= t <= pulse_duration else 0
            h = [hamiltonian, pulse_coeff]
        else:
            h = hamiltonian  # For free evolution, the Hamiltonian is time-independent

        result = mesolve(h, initial_state, np.linspace(0, duration, 100), [], [])
        return result.states[-1]


    def calculate_photon_count(self, state):
        # Calculate the probability of being in |ms=0>
        prob_ms0 = abs(state.overlap(self.ms0_state))**2
        # Scale the probability to get an estimated photon count
        photon_count = prob_ms0 * self.photon_scale_factor
        return photon_count


    def run_simulation(self, evolution_times, hahn_echo=False):
        pulse_sequence = self.generate_pulse_sequence(evolution_times, hahn_echo)
        photon_counts = np.zeros_like(evolution_times)  # Match the size of evolution_times

        for i, tau in enumerate(evolution_times):
            state = self.initial_state  # Reset the state for each new evolution time
            for pulse in pulse_sequence[i*3:(i+1)*3]:  # Only take the relevant pulses for this tau
                if pulse["type"] == "free_evolution":
                    hamiltonian = self.free_hamiltonian
                    state = self.simulate_evolution(state, hamiltonian, pulse["duration"], pulse["type"])
                elif pulse["type"] in ["pi", "pi2"]:
                    hamiltonian = self.pi_pulse_hamiltonian if pulse["type"] == "pi" else self.pi2_pulse_hamiltonian
                    state = self.simulate_evolution(state, hamiltonian, pulse["duration"], pulse["type"])

            # Measure fluorescence after the second pi/2 pulse
            photon_counts[i] = self.calculate_photon_count(state)

        return photon_counts
# Example usage
MW_freq_range = np.linspace(2840, 2900, 250)  # Adjust range and points as needed
B0 = 0 # Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0  # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = 0, 0  # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 6  # Linewidth of the transitions (in MHz)

Bvector_list = vec.getAllframesCartesian(B0, thetaB, phiB)
Evector_list = vec.getAllframesCartesian(E0, thetaE, phiE)
MWvector_list = vec.getAllframesCartesian(1, thetaMW, phiMW)

fluorescence = np.zeros(250)
tau_range = np.linspace(1e-9, 10e-6, 250)  # Up to one period of the Rabi oscillation

for MWvec, Bvec, Evec in zip(MWvector_list, Bvector_list, Evector_list):
    pi2_pulse_duration = (np.pi/2) / (2.87e9 * 2 * np.pi)
    pi_pulse_duration = np.pi / (2.87e9 * 2 * np.pi)
    
    # Adjust initial_state and ms0_state for the 9-dimensional Hilbert space
    initial_state = tensor(basis(3, 1), basis(3, 1))  # This is a pure state in a 9-dimensional space
    initial_state = initial_state * initial_state.dag()  # Convert to density matrix


    ms0_state = tensor(basis(3, 0) * basis(3, 0).dag(), qeye(3))  # This projects onto ms=0 for the electron spin and is identity for the nuclear spin
    
    photon_scale_factor = 100
    free_hamiltonian = snvh.free_hamiltonian(Bvec, Evec)
    mw_hamiltonian = snvh.GShamiltonianMWint(MWvec)

    ramsey_sim = RamseySimulation(pi2_pulse_duration, initial_state, ms0_state, photon_scale_factor, free_hamiltonian, pi_pulse_duration, mw_hamiltonian, mw_hamiltonian)
    fluorescence += ramsey_sim.run_simulation(tau_range, hahn_echo=False)

print(fluorescence)

plt.style.use("classic")
plt.plot(tau_range, fluorescence)
plt.xlabel('Free Precession Time (s)')
plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
plt.title('Simulated Ramsey Fringes')
plt.show()
