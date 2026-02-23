import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as kB, e, m_e, epsilon_0, pi
from scipy.special import wofz
import scipy.constants as c
import isr_test_par as itp

#
# Chapter 3 and Appendix D
# Froula et.al., Plasma Scattering of Electromagnetic Radiation (Second Edition), 2011
# 
def Z(z):
    return 1j * np.sqrt(np.pi) * wofz(z)

def susceptibility(n, T, m, k_radar, omega):
    vth = np.sqrt(2 * kB * T / m)
    lambda_D2 = epsilon_0 * kB * T / (n * e**2)
    
    zeta = omega / (k_radar * vth)
    chi = (1.0 / (k_radar**2 * lambda_D2)) * (1 + zeta * Z(zeta))
    
    return chi, vth, lambda_D2

def thermal_driver(n, T, m, k_radar, omega):
    """
    Returns the fluctuation source:
        (2 n k T / ω) Im{χ}
    including exact ω→0 limit.
    """
    chi, vth, lambda_D2 = susceptibility(n, T, m, k_radar, omega)
    
    F = np.zeros_like(omega, dtype=float)
    
    small = np.abs(omega) < 1e-2
    large = ~small
    
    # Regular frequencies
    F[large] = (2 * n * kB * T / omega[large]) * np.imag(chi[large])
    
    # Analytic ω→0 limit
    limit = np.sqrt(np.pi) / (k_radar**3 * lambda_D2 * vth)
    F[small] = 2 * n * kB * T * limit
    
    return F

def isr_spectrum(freq, k_radar,
                 ne, Te,
                 Ti1,
                 Ti2,
                 frac1,
                 mass1=16,mass2=32):
    
    # this will only work for the ion-line, not including the electron component of the
    # plasma-line spectrum
    
    amu = 1.66053906660e-27
    m_1  = mass1 * amu
    m_2 = mass2 * amu
    
    ni = ne
    n_1 = frac1 * ni
    n_2  = (1 - frac1) * ni
    
    omega = 2 * pi * freq
    
    # --- susceptibilities (for dielectric response)
    chi_e, _, _  = susceptibility(ne, Te, m_e, k_radar, omega)
    chi_i1, _, _  = susceptibility(n_1,  Ti1,  m_1,  k_radar, omega)
    chi_i2, _, _ = susceptibility(n_2, Ti2, m_2, k_radar, omega)
    
    epsilon = 1 + chi_e + chi_i1 + chi_i2
    
    # --- thermal fluctuation sources (THIS includes electron currents)
    Se  = thermal_driver(ne,  Te,  m_e,  k_radar, omega)
    Si1  = thermal_driver(n_1, Ti1, m_1, k_radar, omega)
    Si2 = thermal_driver(n_2,Ti2,m_2,k_radar, omega)
    
    S = (Se + Si1 + Si2) / np.abs(epsilon)**2
    
    spec=np.real(S)
    # remove DC component (weak electron contribution outside ion line)
    spec=(spec-spec[-1])
    
    return spec/np.max(spec)


import matplotlib.pyplot as plt
import isr_spec as isp
from ISRSpectrum import Specinit
radar_freq=440.2e6
for te_ti in itp.te_ti:
    for ti in itp.ti:
        for fr1 in itp.ion_fractions:
            for masses in itp.masses:
                mass1=masses[0]
                mass2=masses[1]

                nes=itp.ne
                Te = te_ti*ti      # K
                Ti = ti
                lambda_radar = c.c/radar_freq
                k_radar = 4 * np.pi / lambda_radar
                freq=itp.freq+1e-6
#                freq = np.linspace(-100e3, 100e3, 4001)+1e-6  # Hz
                
                plt.figure(figsize=(7,5))
                plt.subplot(131)

                for ne in nes:
                    ni = ne
                    S = isr_spectrum(freq, k_radar, ne, Te, Ti, Ti, frac1=fr1,
                                     mass1=mass1,
                                     mass2=mass2)
                    #    S = isr_ionline_single(freq, k_radar, ne, Te, ni, Ti, m_O)
                    plt.semilogy(freq/1e3, S, lw=2,label="$n_e$=%g"%(ne))
    
                    plt.xlabel("Frequency (kHz)")
                    plt.ylabel("Normalized Power")
                plt.title(r"Sheffield ($T_e/T_i=%1.1f\,T_i=%d\,m_1=%d\,m_2=%d\,fr1=%1.1f$)"%(te_ti,ti,mass1,mass2,fr1))
                plt.legend()
                plt.grid(True)
                plt.xlim([np.min(freq)/1e3,np.max(freq)/1e3])
                plt.ylim([1e-3,2])
                # gordeyev
                plt.subplot(132)
                for ne in nes:
                    plpar={"t_i":[Ti,Ti],
                           "t_e":Te,
                           "m_i":[mass1,mass2],
                           "n_e":ne,
                           "freq":radar_freq,
                           "B":45000e-9,
                           "alpha":45, 
                           "ion_fractions":[fr1,1-fr1]}
                    om=2*np.pi*freq
                    il_spec=isp.isr_spectrum(om,plpar=plpar,n_points=2e4,ni_points=2e4,include_electron_component=True)
                    # dc remove for ion line
#                    il_spec=(il_spec-il_spec[-1])
                    il_spec=il_spec#/np.max(il_spec)
                    plt.semilogy(om/2/np.pi/1e3,il_spec)
                plt.grid()

                plt.xlim([np.min(freq)/1e3,np.max(freq)/1e3])
 #               plt.ylim([1e-3,2])
                plt.title(r"K-M (J.V.) ($f=%1.2f MHz$)"%(radar_freq/1e6))
                plt.subplot(133)
                
                ISS2 = Specinit(centerFrequency = 440.2*1e6, bMag = 45000e-9, nspec=2*512, sampfreq=2e6,dFlag=True)
                for ne in nes:
                    datablock90 = np.array([[fr1*ne,Ti,0,1,mass1,0],[(1-fr1)*ne,Ti,0,1,mass2,0],[ne,Te,0,1,1,0]])
                    (om,il_spec,rcs) = ISS2.getspec(datablock90, rcsflag = True, alphadeg=45)
                    # dc remove for ion line
 #                   il_spec=(il_spec-il_spec[-1])
                    plt.semilogy(om/1e3,il_spec)#/np.max(il_spec))
    
                plt.xlim([np.min(freq)/1e3,np.max(freq)/1e3])
#                plt.ylim([1e-3,2])
                plt.grid()
                plt.title("K-M (J.S.)")
                plt.show()
