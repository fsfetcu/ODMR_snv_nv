#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
File to run NV simulations.
"""



import numpy as np
from utils import plotting
from measurements import cwODMR, pulsedODMR
from vectors import vec
import matplotlib.pyplot as plt

#constants
MW_freq_range = np.linspace(2820, 2920, 1000)  # Adjust range and points as needed
B0 = 10 # Magnetic field strength in G
thetaB, phiB = np.pi, np.pi/2  # Direction of the magnetic field in spherical coordinates
E0 = 0  # Electric field strength (assuming no electric field for simplicity)
thetaE, phiE = np.pi/4, np.pi/2  # Direction of the electric field (not relevant here)

# Define MW field parameters (assuming it's perpendicular to the NV axis)
thetaMW, phiMW = 0, np.pi/2 # Direction of MW field
Linewidth = 2.5 # Linewidth of the transitions (in MHz)

Bvec = vec.getAllframesCartesian(B0, thetaB, phiB)[0]
Evec = vec.getAllframesCartesian(E0, thetaE, phiE)[0]
MWvec = vec.getAllframesCartesian(1, thetaMW, phiMW)[0]

cw_odmrDATA = cwODMR.ODMRsingleNV(MW_freq_range,MWvec, Bvec, Evec, Linewidth)
pulsed_odmrDATA = pulsedODMR.pulsedODMRsingleNV(MW_freq_range,MWvec, Bvec, Evec, Linewidth)


simulation = [cw_odmrDATA, pulsed_odmrDATA]#, nvODMR_lock_inDATA, pulsed_odmr_lock_inDATA]
# plotting.plot_ODMR(MW_freq_range, simulation)
l = ["cw", "pulsed"]
plt.figure()
plt.rcParams.update({'font.size': 22})
plt.style.use("classic")
for idx,data in enumerate(simulation):
    data = data/max(data)
    fluoresence = 1 - (data) 
    plt.plot(MW_freq_range, fluoresence, label = f"{l[idx]}")
plt.legend()
plt.xlabel('Microwave frequency (MHz)')
plt.ylabel('Fluoresence (a.u.)')
plt.grid(False)
plt.show()


# Data for different measurements

    ## ODMR data
    # cw_odmrDATA = cwODMR.nvODMR(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)
    # pulsed_odmrDATA = pulsedODMR.pulsednvODMR(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)
    # nvODMR_lock_inDATA = cwODMR.nvODMR_lock_in(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)
    # pulsed_odmr_lock_inDATA = pulsedODMR.pulsednvODMR_lock_in(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)


    # ## Noisy data 
    # noisy_cw_odmrDATA = cwODMR.noisy_nvODMR(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)
    # noisy_pulsed_odmrDATA = pulsedODMR.noisy_pulsednvODMR(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)
    # noisy_nvODMR_lock_inDATA = cwODMR.noisy_nvODMR_lock_in(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)
    # noisy_pulsed_odmr_lock_inDATA = pulsedODMR.noisy_pulsednvODMR_lock_in(MW_freq_range, thetaMW, phiMW, B0, thetaB, phiB,E0, thetaE, phiE, Linewidth)

    # Plotting

    ## ODMR data no noise
    # cw_odmrDATA = cwODMR.ODMRsingleNV(MW_freq_range,MWvec, Bvec, Evec, Linewidth)
    # pulsed_odmrDATA = pulsedODMR.pulsedODMRsingleNV(MW_freq_range,MWvec, Bvec, Evec, Linewidth)
    # simulation = [cw_odmrDATA, pulsed_odmrDATA]#, nvODMR_lock_inDATA, pulsed_odmr_lock_inDATA]
    # plotting.plot_ODMR(MW_freq_range, simulation)

    # ## odmr no noise lock in
    # simulation = [nvODMR_lock_inDATA, pulsed_odmr_lock_inDATA]
    # plotting.plot_ODMR(MW_freq_range, simulation)

    # ## ODMR data with noise
    # simulation = [noisy_cw_odmrDATA, noisy_pulsed_odmrDATA]#
    # plotting.plot_ODMR(MW_freq_range, simulation)

    # Define the range of free precession times
    # tau_range = np.linspace(0, 2 * np.pi / (2.87e9), 1000)  # Up to one period of the Rabi oscillation

    # # Get the simulated fluorescence signal as a function of free precession time
    # fluorescence_signal, tau_range = Ramsey_sequence(MWfreq, Bvec, Evec, Linewidth, tau_range)