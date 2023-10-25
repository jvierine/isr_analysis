# Antistarlink

The peak of the distribution of space objects is now at 550 km altitude. And the number of satellites at this altitude is rather high compared to what the situation was before the introduction of the Starlink fleet. This means a lot of heacache for ionospheric radar measurements.

Ionospheric radar measurements also suffer from more and more man made radio interference. The Millstone Hill radar is perhaps one of the most challenging radar sites when it comes to inteference. There are lots of radio transmitters operating near the radar frequency (440.2 MHz), there are some radars nearby that jam the receiver, and there is ample power line interference, generating spikes everywhere within the receiver signal. 

Code in this repo is a lag-profile inversion implementation based on sparse matrices, which aims to mitigate RFI and satellite clutter. This is done by outlier rejection and sample variance based weighting of lagged products. The word of warning: code is not production ready. I'll let you know if we ever get there. 

juha
