import numpy as n
import matplotlib.pyplot as plt
import outlier_lpi as olpi
import fit_lpi as flpi

# we assume that these folders exists under the file path:
# rf_data/misa-l, rf_data/tx-h, rf_data/zenith-l,
# metadata/antenna_control_metadata,  metadata/id_metadata, 
# metadata/powermeter
#
datadir="/media/j/4df2b77b-d2db-4dfa-8b39-7a6bece677ca/eclipse2024/usrp-rx0-r_20240407T100000_20240409T110000/"
# 30 us range gating, matched bit length
rg=30
ch="misa-l"
if False:
    olpi.lpi_files(dirname=datadir,
                   avg_dur=10,  # n seconds to average
                   channel=ch,
                   rg=rg,       # how many microseconds is one range gate
                   min_tx_frac=0.2, # this fraction of the pulse can be missing
                   pass_band=0.018e6, # ion line pass band width
                   filter_len=100,    # short filter, less problems with correlated noise, more problems with spike-like RFI
                   maximum_range_delay=7200,
                   save_acf_images=True,
                   lag_avg=1,
                   reanalyze=True)

flpi.fit_lpifiles(dirn=datadir,
                  channel=ch,
                  postfix="_%d"%(rg),
                  max_dt=300,
                  plot=0,
                  first_lag=0,
                  reanalyze=True,
                  range_avg=n.array([1,3,5]))
