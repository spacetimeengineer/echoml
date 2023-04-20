# -*- coding: utf-8 -*-
"""
EK60-netCDF_view-echograms

Use echopype to read EK60 data in netCDF4 format and view echograms

jech
"""

import echopype as ep
from echopype import open_raw
from pathlib import Path
from dask.distributed import Client
import xarray as xr
import echopype.visualize as epviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the Client gives lots of error messages, so I don't use it for now
#client = Client()

# get a single file
#filename = Path('/home/mjech/Desktop/butterfish/D20090916-T135430.nc')
#dataDirectory = filename.parent

# get multiple files
dataDirectory = Path('/home/mryan/Desktop/Ryan_Mike/src/butterfish/netCDF4_Files')
filelist = []
for f in dataDirectory.glob('**/*.nc'):
    print('.nc file: ', f)
    filelist.append(f)

#filelist = [filename]
edlist = []
for f in filelist:
    #edlist = [ep.open_converted(str(filename))]
    edlist = [ep.open_converted(str(f))]

#ed = ep.combine_echodata(edlist, client=client)
ed = ep.combine_echodata(edlist)

# apply calibrations and compute Sv
Sv = ep.calibrate.compute_Sv(ed)

# remove noise
# this only removes background noise
Sv_NR = ep.preprocess.remove_noise(Sv, range_sample_num=30, ping_num=5, 
                                   SNR_threshold=6)
# compute MVBS
#Sv_MVBS = ep.preprocess.compute_MVBS(Sv_NR, range_meter_bin=1, 
#                                     ping_time_bin='20S')

# swap channel dimension for frequency
Sv = ep.preprocess.swap_dims_channel_frequency(Sv)
Sv.Sv.sel(frequency_nominal=120000).plot.pcolormesh(x='ping_time', 
               cmap='viridis', vmin=-80, vmax=-30)

Sv_NR = ep.preprocess.swap_dims_channel_frequency(Sv_NR)
Sv_NR.Sv.sel(frequency_nominal=120000).plot.pcolormesh(x='ping_time', 
               cmap='viridis', vmin=-80, vmax=-30)

# get xarray object for selected frequency
SvNR_120kHz = Sv_NR.Sv.sel(frequency_nominal=120000)
# if you want just the array as a numpy array
#SvNR_120kHz = Sv_NR.Sv.sel(frequency_nominal=120000).values


# get a list of frequencies
flist = Sv.frequency_nominal.values

# save processed data
#Sv_NR.to_netcdf('file name')


