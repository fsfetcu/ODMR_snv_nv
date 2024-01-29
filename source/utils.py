#!usr/bin/python3
# -*- coding: utf-8 -*-

""""
This module contains generic useful functions
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip import qeye, tensor, basis

class math_functions:

    """
    A module for mathematical functions.
    """

    @staticmethod
    def lorentzian(x, x0, fwhm):
        """
        Lorentzian function.

        Parameters
        ----------
        x : numpy.ndarray
            x values.
        x0 : float
            Center of the Lorentzian.
        fwhm : float
            Full width at half maximum of the Lorentzian.

        Returns
        -------
        numpy.ndarray
            Lorentzian function.
        """

        return 1 / (1 + (x - x0)**2 / (fwhm / 2)**2)

    @staticmethod
    def dispersive_lorentzian(x, x0, fwhm):
        """
        Dispersive lineshape function.

        Parameters
        ----------
        x : numpy.ndarray
            x values.
        x0 : float
            Center of the Lorentzian.
        fwhm : float
            Full width at half maximum of the Lorentzian.
        
        Returns
        -------
        numpy.ndarray
            Dispersive lineshape function.
        """

        xred = (x - x0) / (fwhm / 2)
        return -2 * xred / (1 + xred**2)**2

    @staticmethod

    def decay_envelope(tau, frequency, T2_star, phi):
        """
        Calculate the decay envelope for the Ramsey fringes.

        Parameters
        ----------
        tau : numpy.ndarray
            Free precession time.
        A0 : float
            Initial amplitude of the signal.
        frequency : float
            Frequency of the oscillations.
        T2_star : float
            Dephasing time.
        phi : float
            Initial phase of the oscillation.

        Returns
        -------
        numpy.ndarray
            Decay envelope for the Ramsey fringes.
        """
        return np.cos(2 * np.pi * frequency * tau + phi) * np.exp(-tau / T2_star)



class plotting:

    """
    A module for plotting.
    """
    
    @staticmethod
    def plot_ODMR(MWfreq, simulation):
        
        """
        Plots the ODMR fluoresence spectrum.

        Parameters
        ----------
        MWfreq : numpy.ndarray
            Array of frequencies.
        Tstrength : numpy.ndarray
            Transition strengths array of all simulations.
        """

        PL_BASELINE = 1.5e6  # Adjust this value as needed for your simulation
        PL_DROP = 0.1 * PL_BASELINE

        # Find peaks 
        #from scipy.signal import find_peaks
        # peakyboo, _ = find_peaks(-fluoresence, prominence=0.005) 
        # freq1 = max(MWfreq[peakyboo])
        # freq2 = min(MWfreq[peakyboo])
        # l = ["CW", "Pulsed"]
        # Plot the spectrum
        plt.figure(figsize=(8, 6))
        plt.style.use("classic")
        for idx,data in enumerate(simulation):
            data = data/np.max(data)
            fluoresence = 1 - (data) 
            plt.plot(MWfreq, fluoresence)#, label = f"{l[idx]}")
            

            # # Plot FWHM arrow
            # if idx == 1:
            #     half_max = np.max(fluoresence) / 2
            #     # Find where the data first drops below half the max
            #     left_idx = np.where(fluoresence <= half_max)[0][0]
            #     # Find where the data last rises above half the max
            #     right_idx = np.where(fluoresence <= half_max)[0][-1]
            #     # These indices can be used to find the frequency values for the FWHM
            #     fwhm_left = MWfreq[left_idx]
            #     fwhm_right = MWfreq[right_idx]
            #     plt.annotate('', 
            #                 xy=(fwhm_left, half_max), xycoords='data',
            #                 xytext=(fwhm_right, half_max), textcoords='data',
            #                 arrowprops=dict(arrowstyle="<->", color='black', ls = '-',	linewidth=1.0))
            #     plt.text((fwhm_left + fwhm_right) / 2, half_max, 'FWHM', 
            #         horizontalalignment='center', verticalalignment='bottom')

        plt.xlabel('Microwave frequency (MHz)', fontsize=22)
        plt.ylabel('Fluoresence (a.u.)', fontsize=22)
        # plt.gca().axes.yaxis.set_ticklabels([])
        # plt.gca().axes.xaxis.set_ticklabels([])
        # # reduce opacity by 50% for axvline
        # # Draw the first segment of the dotted line
        # plt.axvline(x=2870, ymin=0, ymax=(half_max)-0.02, color='r', linestyle='--')

        # # Draw the second segment of the dotted line
        # plt.axvline(x=2870, ymin=(half_max)+0.05, ymax=1, color='r', linestyle='--',label = 'Resonance frequency')

        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #   ncol=3, fancybox=True, shadow=True)
        plt.grid(False)
        plt.show()

    def ple_snv(MWfreq, simulation):
        plt.figure()
        plt.style.use("classic")
        for idx,data in enumerate(simulation):
            data = data/np.max(data)
            plt.plot(MWfreq, simulation, label = f"{idx}")
            #check if the data in sumulation has same values as each other
        # plt.plot(MWfreq[peakyboo], fluoresence[peakyboo]/max(fluoresence), "xD", label='Peak value(s)')
        plt.xlabel('Microwave frequency (MHz)')
        plt.ylabel('Fluoresence intensity (a.u.)')
        plt.legend()
        plt.grid(True)
        plt.show()


    @staticmethod
    #TODO
    def plot_ramsey(MWfreq, simulation):
        PL_BASELINE = 1.5e6  # Adjust this value as needed for your simulation
        PL_DROP = 0.1 * PL_BASELINE
        
        # Plot the Ramsey fringes (fluorescence vs free precession time)
        plt.figure()
        plt.style.use("classic")
        # data = data/np.max(data)
        for idx,data in enumerate(simulation):
            fluoresence = PL_BASELINE - (data) * PL_DROP
            plt.plot(MWfreq, data, 'o-', markersize=1, label = f"{idx}")
        plt.xlabel('Free Precession Time (us)')
        plt.ylabel('Fluorescence Intensity (Arbitrary Units)')
        plt.title('Simulated Ramsey Fringes')
        plt.legend()
        plt.show()


class noise:
    
    """
    A module for adding different kind of noise to signals.
    """

    @staticmethod
    def NoiseOf(signal, noise_type, NoisePercentage):
        
        """
        Add 'random' noise to a signal. 
        The noise is Gaussian with a standard deviation of x% of the maximum value of the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            Array of values.
        noise_type : str
            Type of noise to add. Options :{'gaussian','poisson','uniform'}.
            Default is 'gaussian'.
        NoisePercentage : float
            Percentage of the maximum value of the signal to use as standard deviation.
            Default is 0.02.
        
        Returns
        -------
        numpy.ndarray
            Noisy signal.
        """

        noise_level = NoisePercentage * np.max(signal)

        if noise_type == 'gaussian':
            return np.random.normal(0, noise_level, signal.shape)

        elif noise_type == 'poisson':
            return np.random.poisson(0, noise_level, signal.shape)

        elif noise_type == 'uniform':
            return np.random.uniform(-noise_level, noise_level, signal.shape)

        else:
            return np.random.normal(0, noise_level, signal.shape)


class operators:
    
    @staticmethod
    def square_pulse(amplitude,duration):
        """
        Generate a square pulse.

        Parameters:
        amplitude (float): Amplitude of the pulse.
        duration (float): Duration of the pulse in seconds.

        Returns:
        numpy.ndarray: Time-dependent amplitude of the pulse.
        """
        return np.ones(int(duration)) * amplitude
