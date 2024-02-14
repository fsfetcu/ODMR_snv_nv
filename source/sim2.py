import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR
from vectors import vec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from hamiltonian import SingleSnVHamiltonian

def twod():
    # Assuming SingleSnVHamiltonian class is already defined as in your code snippet

    # Constants
    B_magnitude = 0.1  # Replace with the actual magnitude of the magnetic field

    # Grid definition
    phi_values = np.linspace(0, 2*np.pi, 50)  # Azimuthal angle from 0 to 2pi
    theta_values = np.linspace(0, np.pi, 50)   # Polar angle from 0 to pi
    qubit_freqs = np.zeros((len(theta_values), len(phi_values)))

    # Calculate qubit frequency for each (phi, theta)
    for i, theta in enumerate(theta_values):
        for j, phi in enumerate(phi_values):
            # Convert spherical to Cartesian coordinates
            B = vec.getAllframesCartesian(B_magnitude, theta, phi)[0]

            # Calculate eigenenergies
            E_I, _ = SingleSnVHamiltonian.SNV_eigenEnergiesStates(B)
            
            # Assuming the qubit frequency is the difference between the first two eigenenergies
            qubit_freq = E_I[1] - E_I[0]
            qubit_freqs[i, j] = qubit_freq

    # Create the 2D plot
    plt.figure(figsize=(10, 5))
    plt.style.use("classic")
    phi_grid, theta_grid = np.meshgrid(phi_values, theta_values)
    plt.contourf(phi_grid, theta_grid, qubit_freqs, 100, cmap='viridis')
    plt.colorbar(label='Qubit Frequency (Hz)')
    plt.xlabel('$\\phi_B$ (rad)')
    plt.ylabel('$\\theta_B$ (rad)')
    plt.title('Qubit Frequency as a Function of Magnetic Field Direction')
    plt.show()
    pass

def oned():
        # Assuming SingleSnVHamiltonian class is already defined as in your code snippet

    # Constants
    B_magnitude = 0.1  # Replace with the actual magnitude of the magnetic field

    # Grid definition
    phi_values = np.linspace(0, 2*np.pi, 1000)  # Azimuthal angle from 0 to 2pi
    qubit_freqs = np.zeros(len(phi_values))

    # Calculate qubit frequency for each (phi, theta)
    for i, phi in enumerate(phi_values):
        # Convert spherical to Cartesian coordinates
        B = vec.getAllframesCartesian(B_magnitude, np.pi/2, phi)[0]

        # Calculate eigenenergies
        E_I, _ = SingleSnVHamiltonian.SNV_eigenEnergiesStates(B)
        
        # Assuming the qubit frequency is the difference between the first two eigenenergies
        qubit_freq = E_I[1] - E_I[0]
        qubit_freqs[i] = qubit_freq

    # Create the 2D plot
    plt.figure(figsize=(8, 6))
    plt.style.use("classic")
    plt.plot(phi_values, qubit_freqs/1e9)
    plt.xlabel('$\\phi_B$ (rad)')
    plt.ylabel('Qubit Frequency (GHz)')
    plt.show()

def bd():
    # Constants
    B_magnitude = np.linspace(0,0.5,1000) # Replace with the actual magnitude of the magnetic field

    # Grid definition
    phi_values = np.pi/1.5  # Azimuthal angle from 0 to 2pi
    qubit_freqs = np.zeros(len(B_magnitude))

    # Calculate qubit frequency for each (phi, theta)
    for i, B in enumerate(B_magnitude):
        # Convert spherical to Cartesian coordinates
        B = vec.getAllframesCartesian(B, np.pi/2, phi_values)[0]

        # Calculate eigenenergies
        E_I, _ = SingleSnVHamiltonian.SNV_eigenEnergiesStates(B)
        
        # Assuming the qubit frequency is the difference between the first two eigenenergies
        qubit_freq = E_I[1] - E_I[0]
        qubit_freqs[i] = qubit_freq

    # Create the 2D plot
    plt.figure(figsize=(8, 6))
    plt.style.use("classic")
    plt.plot(B_magnitude, qubit_freqs)
    plt.xlabel('Magnetic field strength (T)')
    plt.ylabel('Qubit Frequency (Hz)')
    plt.show()

def strain():
    


    # 2d plot, qubit freq vs strain and phi

    # Constants
    B_magnitude = 0.1  # Replace with the actual magnitude of the magnetic field
    strain_values = np.linspace(0, 300e9, 100)  # Strain values to explore
    phi_values = np.linspace(0, 2*np.pi, 100)  # Azimuthal angle from 0 to 2pi

    # Calculate qubit frequency for each (phi, strain)
    qubit_freqs = np.zeros((len(strain_values), len(phi_values)))
    for i, strain in enumerate(strain_values):
        for j, phi in enumerate(phi_values):
            # Calculate eigenenergies
            B = vec.getAllframesCartesian(B_magnitude, np.pi/2, phi)[0]
            E_I, _ = SingleSnVHamiltonian.SNV_eigenEnergiesStates_strain(B, strain)
            
            # Assuming the qubit frequency is the difference between the first two eigenenergies
            qubit_freq = E_I[1] - E_I[0]
            qubit_freqs[i, j] = qubit_freq

            print(i,j)
    # Create the 2D plot
    plt.figure(figsize=(10, 5))
    plt.style.use("classic")
    phi_grid, strain_grid = np.meshgrid(phi_values, strain_values)
    plt.contourf(phi_grid, strain_grid, qubit_freqs, 100, cmap='viridis')
    plt.colorbar(label='Qubit Frequency (Hz)')
    plt.xlabel('$\\phi_{MW}$ (rad)')
    plt.ylabel('Ground strain (Hz)')
    plt.show()
strain()