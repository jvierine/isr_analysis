import outlier_lpi as olpi
import avg_range_doppler_spec as ards
import traceback

#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03a/usrp-rx0-r_20211203T224500_20211204T160000/",
#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-01/usrp-rx0-r_20211201T230000_20211202T160100/",
#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-05/usrp-rx0-r_20230905T214448_20230906T040054",
#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-24/usrp-rx0-r_20230924T200050_20230925T041059/",
#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533/",
#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-03/usrp-rx0-r_20211203T000000_20211203T033600/",
 #     "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-05/usrp-rx0-r_20211205T000000_20211205T160100/",
  #    "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-06/usrp-rx0-r_20211206T000000_20211206T132500/",
   #   "/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2021-12-21/usrp-rx0-r_20211221T125500_20211221T220000/",
#"/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-28/usrp-rx0-r_20230928T211929_20230929T040533/"
#dirs=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-09-20/usrp-rx0-r_20230920T202127_20230921T040637/"]
dirs=["/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/isr/2023-10-14/usrp-rx0-r_20231014T130000_20231015T041500"]
for d in dirs:
    try:
        olpi.lpi_files(dirname=d,
                       avg_dur=10,  # n seconds to average
                       channel="zenith-l2",
                       rg=30,       # how many microseconds is one range gate
                       min_tx_frac=0.5, # of the pulse can be missing
                       pass_band=0.018e6, # +/- 50 kHz 
                       filter_len=100,    # short filter, less problems with correlated noise, more problems with RFI
                       maximum_range_delay=7200,
                       save_acf_images=True,
                       lag_avg=1,
                       reanalyze=False)
    except:
        traceback.print_exc()
    try:
        olpi.lpi_files(dirname=d,
                       avg_dur=10,  # n seconds to average
                       channel="zenith-l",
                       rg=30,       # how many microseconds is one range gate
                       min_tx_frac=0.5, # of the pulse can be missing
                       pass_band=0.018e6, # +/- 50 kHz 
                       filter_len=100,    # short filter, less problems with correlated noise, more problems with RFI
                       maximum_range_delay=7200,
                       save_acf_images=True,
                       lag_avg=1,
                       reanalyze=False)
    except:
        traceback.print_exc()
        
    try:
        olpi.lpi_files(dirname=d,
                       avg_dur=10,  # n seconds to average
                       channel="misa-l",
                       rg=30,       # how many microseconds is one range gate
                       min_tx_frac=0.5, # of the pulse can be missing
                       pass_band=0.018e6, # +/- 50 kHz 
                       filter_len=100,    # short filter, less problems with correlated noise, more problems with RFI
                       maximum_range_delay=7200,
                       save_acf_images=True,
                       lag_avg=1,
                       reanalyze=False)
    except:
        traceback.print_exc()
    
    try:
        ards.avg_range_doppler_spectra(dirname=d,
                                       channel="zenith-l",
                                       save_png=True,
                                       avg_dur=10,
                                       reanalyze=False
                                       )
    except:
        traceback.print_exc()
    try:
        ards.avg_range_doppler_spectra(dirname=d,
                                       channel="misa-l",
                                       save_png=True,
                                       avg_dur=10,
                                       reanalyze=False
                                       )
    except:
        traceback.print_exc()
    
    
