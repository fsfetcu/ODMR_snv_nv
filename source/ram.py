import numpy as np
from utils import plotting, math_functions
from measurements import cwODMR, pulsedODMR
import matplotlib.pyplot as plt
from vectors import vec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
#constants
MW_freq_range = np.linspace(2840, 2900, 500)  # Adjust range and points as needed
B0 = 0# Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0 # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = np.pi / 2, 0   # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 1  # Linewidth of the transitions (in MHz)

data = []

MWvec_x = [1, 0, 0]

# Linear polarization along y-axis
MWvec_y = [0, 1, 0]

# Right circular polarization (in xy-plane)
MWvec_rcp = [1/np.sqrt(2), 1j/np.sqrt(2), 0]

# Left circular polarization (in xy-plane)
MWvec_lcp = [1/np.sqrt(2), -1j/np.sqrt(2), 0]

MWvec = [MWvec_x, MWvec_y, MWvec_rcp, MWvec_lcp]
Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
Evec = vec.getAllframesCartesian(E0, thetaE, phiE)[0]

# Define the range of free precession times



# fluorescence_signal_normalized = fluorescence_signal / max(fluorescence_signal)

# # Plotting the normalized fluorescence signal
# plt.plot(tau_range, fluorescence_signal, 'o', color='blue', markersize=3)
# plt.xlabel('Free Precession Time (s)')
# plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
# plt.title('Simulated Ramsey Fringes')
# plt.show()

# fluorescence_signal = pulsedODMR.ramsey_spectroscopy(tau_range, MWvec[3],Bvec, Evec, Linewidth)
# # Bvec = vec.getAllframesCartesian(100, thetaB, phiB)[0]
# # fluorescence_signal2 = pulsedODMR.ramsey_spectroscopy(tau_range, MWvec[3],Bvec, Evec, Linewidth)
# # data = [fluorescence_signal, fluorescence_signal2]
# plotting.plot_ramsey(tau_range, [fluorescence_signal])   

def exp_decay(x, a, b, c):
    return a*np.exp(-x/b) + c

tau_range = np.linspace(1e-10,20,500)#np.arange(1e-6, 0.2, 1/2870)  # Up to one period of the Rabi oscillation
T1 = np.linspace(1,40,200)
# tlist , result, T2 = pulsedODMR.ram2(tau_range,MWvec[3],Bvec,Evec,Linewidth)#(tau_range,MWvec[3],Bvec, Evec, Linewidth)

probabilities = np.array([pulsedODMR.relaxation(tau, T1, Linewidth) for tau in tau_range])

# Normalize probabilities
probabilities /= probabilities.max()


# Assuming `tau_range` is your array of time points and `probabilities` is your oscillation data
peaks, _ = find_peaks(probabilities,prominence=0.01)
peak_times = tau_range[peaks]
peak_values = probabilities[peaks]

# Fit the exponential decay to the peaks
params, _ = curve_fit(exp_decay, peak_times, peak_values, p0=[1, 1/(np.pi*Linewidth), 0.5])
print("T2 = ", params[1])
# Generate the envelope using the fitted parameters
envelope = exp_decay(tau_range, *params)

# peaks, _ = find_peaks(result[1],prominence=0.01)
# peak_times = tau_range[peaks]
# peak_values = result[1][peaks]

# # Fit the exponential decay to the peaks
# params, _ = curve_fit(exp_decay, peak_times, peak_values,p0=[1, T2, 0.5])

# # Generate the envelope using the fitted parameters
# envelope1 = exp_decay(tau_range, *params)


# print("T2* = ",params[1])


T2 = 1/(np.pi*Linewidth)
plt.figure()
plt.style.use("classic")
# plt.plot(tlist[:-1], result[1][:-1], '-', linewidth = 3, label="Ramsey signal") # result[1][:-1]
# plt.plot(tlist[:-1], (np.exp(-1./T2*tlist[:-1]))*0.5 + 0.5, color = "r",label="Exponential envelope (Ramsey)")
# plt.plot(tlist[:-1], -np.exp(-1./T2*tlist[:-1])*0.5 +0.5, color = "r")
# plt.plot(tau_range, probabilities, 'o-',linewidth = 1, label='Hahn Echo')
# plt.plot(tau_range, envelope, 'y-', label='Exponential envelope (Hahn)')
plt.plot(T1, probabilities, 'o-',linewidth = 1, label='T1')
# plt.plot(tau_range, envelope1, 'r-', label='Exponential envelope (Ramsey)')

plt.xscale('linear')
plt.xlabel('Free evolution time $\\tau$ ($\\mu s$)')
plt.ylabel('Population of |$0\\rangle$')
plt.legend()
plt.show()

