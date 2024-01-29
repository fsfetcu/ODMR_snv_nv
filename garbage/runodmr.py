import numpy as np
import matplotlib.pyplot as plt
import __odmr as simnv
from scipy.signal import find_peaks

# Define MW frequency range (in MHz) for the ODMR spectrum
MW_freq_range = np.linspace(2840, 2900, 500)  # Adjust range and points as needed

# Define the parameters for the NV center environment
# Example values: Magnetic field (B0), electric field (E0), etc.
B0 = 4 # Magnetic field strength in G
thetaB, phiB = np.pi / 2, 0  # Direction of the magnetic field in spherical coordinates
E0 = 0  # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = 0, 0  # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = np.pi/2 , 0  # Direction of MW field
Linewidth = 6  # Linewidth of the transitions (in MHz)

# Calculate the ODMR spectrum
ODMR_spectrum = simnv.ESR_NVensemble_qutip_noisy(MW_freq_range, thetaMW, phiMW, 
                                   B0, thetaB, phiB, 
                                   E0, thetaE, phiE, 
                                   Linewidth)

                                
data = ODMR_spectrum/np.max(ODMR_spectrum) 


ODMR_spectrum = simnv.pulsed_ESR_NVensemble_qutip_noisy(MW_freq_range, thetaMW, phiMW, 
                                   B0, thetaB, phiB, 
                                   E0, thetaE, phiE, 
                                   Linewidth)

data2 = ODMR_spectrum/np.max(ODMR_spectrum) 

# ODMR_spectrum = simnv.ESR_Lockin_singleNV_qutip(MW_freq_range, MWvec, Bvec, Evec, Linewidth)


# Assume a baseline PL count when no microwaves are applied (e.g., 500k counts)
# Assume a baseline PL count when no microwaves are applied (e.g., 500k counts)
PL_baseline = 1.5e6  # Adjust this value as needed for your simulation

# Calculate the decrease in PL count at resonance (10% drop)
PL_drop = 0.1 * PL_baseline

# Calculate the PL intensity by subtracting the transition strength (scaled to the drop) from the baseline
# The transition strength is scaled so that at maximum it leads to a 10% drop, i.e., to 90% of the baseline
PL_intensity = PL_baseline - (data) * PL_drop
PL_intensity2 = PL_baseline - (data2) * PL_drop


# Now, PL_intensity should show a 10% dip at the resonant frequencies


peaks, _ = find_peaks(-PL_intensity/max(PL_intensity), prominence=0.01)



# fplus = max(MW_freq_range[peaks])
# fminus = min(MW_freq_range[peaks])


# def magneticfieldstrength(fplus,fminus):
#     gamma_e = 28.024e9
#     return (fplus-fminus)/(2*gamma_e)

# MFS = magneticfieldstrength(fplus,fminus)

# Plot the PL intensity as a function of microwave frequency
plt.figure(figsize=(10, 6),layout='constrained')
plt.style.use("classic")
plt.plot(MW_freq_range, PL_intensity/max(PL_intensity), label = "CW-odmr")
plt.plot(MW_freq_range, PL_intensity2/max(PL_intensity2), label='Pulsed-odmr')
# plt.plot(MW_freq_range[peaks], PL_intensity[peaks]/max(PL_intensity), "hr", label='Peak value(s)')
plt.xlabel('Microwave Frequency (MHz)')
plt.ylabel('Photoluminescence Intensity (Counts)')
plt.title('Simulated ODMR Spectrum of NV Center')
plt.legend()
plt.grid(True)
plt.show()

