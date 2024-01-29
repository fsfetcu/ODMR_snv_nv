from hamiltonian import singleNVhamiltonian
import numpy as np
import matplotlib.pyplot as plt

# Define a range for theta_B and phi_B
theta_B_values = np.linspace(0, np.pi, 100)  # from 0 to pi
phi_B_values = np.linspace(0, 2*np.pi, 100)  # from 0 to 2*pi

# Define the magnetic field magnitude
B_magnitude = 1  # Replace with the actual magnitude needed

# Placeholder for the eigenenergies
eigen_energies = np.zeros((len(theta_B_values), len(phi_B_values)))

# Calculate the eigenenergies for each orientation
for i, theta_B in enumerate(theta_B_values):
    for j, phi_B in enumerate(phi_B_values):
        # Magnetic field vector in spherical coordinates
        B = np.array([
            B_magnitude * np.sin(theta_B) * np.cos(phi_B),  # Bx
            B_magnitude * np.sin(theta_B) * np.sin(phi_B),  # By
            B_magnitude * np.cos(theta_B)                   # Bz
        ])

        # Compute the eigenenergies and eigenstates
        energies, states = singleNVhamiltonian.NV_eigenEnergiesStates(B,E = np.array([0,0,0]))

        # Store the energy corresponding to B_parallel (which is Bz here)
        # Replace '0' with the appropriate index for the energy level of interest
        eigen_energies[i, j] = energies[0]
        print(i, j)
# Create a meshgrid for plotting
Theta, Phi = np.meshgrid(theta_B_values, phi_B_values, indexing='ij')

# Plotting the eigenenergies as a contour plot
plt.figure(figsize=(10, 6))
cp = plt.contourf(Phi, Theta, eigen_energies, levels=50, cmap='viridis')
plt.colorbar(cp, label='Frequency difference (MHz)')
plt.xlabel('$\phi_B$ (radians)')
plt.ylabel('$\theta_B$ (radians)')
plt.title('Eigenenergies of NV Center as a Function of Magnetic Field Orientation')
plt.show()