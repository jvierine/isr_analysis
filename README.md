# Plasma-parameter estimation tools (PET)

This repository contains a set of tools for incoherent scatter radar plasma-parameter analysis. Quite many measures are taken in order to mitigate space object contamination and radio interference. There is a lag-profile inversion based on sparse matrix libraries for high SNR measurements that often get you well into the top-side of the ionosphere, and a range-Doppler based fitter for top-side plasma-parameters. 

# Long pulse analysis

Long uncoded pulse are used primarily on the top-side of the ionosphere, where the signal-to-noise ratio is very low. This mode doesn't have very good range resolution (72 km for a 480 microsecond pulse), but it includes a large volume of plasma and hence maximizes the SNR. The analysis is done in range-Doppler domain and it uses a tapered window to reduce spectral leakage from strong out of band radio interference signals, which are often present in the Millstone Hill radar and overpower the ion-line.

There are two main routines that need to be run in sequence: <code>avg_range_doppler_spec.py</code> and <code>fit_lp.py</code>. 

# Coded long pulse analysis

Coded pulses are used primarily on bottom-side of the ionosphere, where the signal-to-noise ratio is high. The only mode supported currently is the 480 microsecond long pulse with 30 microsecond bits. The uncoded long pulse will also be included in the analysis, but the deconvolution of autocorrelation functions is not overdetermined without the coded long pulses. 

There are two main routines that need to be run in sequence: <code>outlier_lpi.py</code> and <code>fit_lpi.py</code>



The word of warning: code is still being developed and tested. 
