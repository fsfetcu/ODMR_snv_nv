# import numpy as np
# import matplotlib.pyplot as plt
# from qutip import *

# # Constants
# max_tau = 20
# min_T1 = 0.2
# max_T1 = 50
# resonant = 2870
# tau_values = np.linspace(0, max_tau, 400)  # Time delays after the pi pulse
# T1_values = np.linspace(min_T1, max_T1, 100)  # Different T1 relaxation times

# # Results matrix
# results_matrix = np.zeros((len(T1_values), len(tau_values)))

# for i, T1 in enumerate(T1_values):
#     H_free = sigmaz() * resonant / 2  # Free Hamiltonian
#     excited_state = basis(2, 1)  # Start in the excited state |1>
#     e_ops = [excited_state * excited_state.dag()]  # Measure the population of the excited state
#     c_ops = [sigmax() * np.sqrt(1/(T1))]  # Collapse operator for T1 relaxation

#     # Solve the master equation for each T1 and tau
#     result = mesolve(H_free, excited_state, tau_values, c_ops=c_ops, e_ops=e_ops)
#     results_matrix[i, :] = result.expect[0]  # Store the population of |1>

# # Plotting
# plt.figure(figsize=(8, 6))
# # Plot the population of the excited state |1> as a function of tau and T1
# plt.imshow(results_matrix, extent=[tau_values.min(), tau_values.max(), T1_values.min(), T1_values.max()], aspect='auto', origin='lower')
# plt.colorbar(label='Population of State |1>')
# plt.xlabel('$\\tau$')
# plt.yscale('linear')
# plt.ylabel('$T_1$')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from qutip import *

# # Constants
# max_tau = 20
# T1 = 10  # Specific T1 value for the simulation
# resonant = 2870
# tau_values = np.linspace(0, max_tau, 400)  # Time delays after the pi pulse

# # Results matrix for T1 = 10
# results = np.zeros(len(tau_values))

# H_free = sigmaz() * resonant / 2  # Free Hamiltonian
# excited_state = basis(2, 1)  # Start in the excited state |1>
# e_ops = [excited_state * excited_state.dag()]  # Measure the population of the excited state
# c_ops = [sigmax() * np.sqrt(1/(T1))]  # Collapse operator for T1 relaxation, should be sigmam()

# # Solve the master equation for T1 = 10 and each tau
# result = mesolve(H_free, excited_state, tau_values, c_ops=c_ops, e_ops=e_ops)
# results = result.expect[0]  # Store the population of |1>

# # Plotting the decay curve
# plt.figure(figsize=(8, 6))
# plt.style.use("classic")
# plt.plot(tau_values, results, label='Population of State |1>')
# plt.xlabel('Tau (Time after pi pulse)')
# plt.ylabel('Population of State |1>')
# plt.title(f'T1 Relaxation for T1 = {T1}')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

# Constants
max_tau = 20
min_T1 = 0.2
max_T1 = 50
resonant = 2870
tau_values = np.linspace(0, max_tau, 400)  # Time delays after the pi pulse
T1_values = np.linspace(min_T1, max_T1, 3000)  # Different T1 relaxation times
num_T1 = len(T1_values)

# Results matrix
results_matrix = np.zeros((num_T1, len(tau_values)))

for i, T1 in enumerate(T1_values):
    H_free = sigmaz() * resonant / 2  # Free Hamiltonian
    excited_state = basis(2, 1)  # Start in the excited state |1>
    e_ops = [excited_state * excited_state.dag()]  # Measure the population of the excited state
    c_ops = [sigmax() * np.sqrt(1/T1)]  # Collapse operator for T1 relaxation
    result = mesolve(H_free, excited_state, tau_values, c_ops=c_ops, e_ops=e_ops)
    results_matrix[i, :] = result.expect[0]  # Store the population of |1>

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Create a color map
norm = Normalize(min_T1, max_T1)
colors = viridis(norm(T1_values))

# Plot each T1 slice with the corresponding color
for i in range(num_T1):
    ax.plot(tau_values, results_matrix[i, :], color=colors[i], lw=2)

# Creating a colorbar with the T1 label
sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib
cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(min_T1, max_T1, 10))
cbar.set_label('$T_1$')

ax.set_xlabel('$\\tau$ ')
ax.set_ylabel('Population of state $|1\\rangle$')

plt.show()