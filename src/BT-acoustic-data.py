
"""
Get acoustic data during bottom trawls

jech
"""

import echopype as ep
from echopype import open_raw
from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap
import matplotlib.pyplot as plt
#from echolab2.plotting.matplotlib import echogram
import echopype.visualize as epviz
from datetime import datetime, timedelta, timezone
import pandas as pd
import json
from aqstxUtilities import seabedDetection as sbd
from aqstxUtilities import lineMask as lm

# get a json file with the trawl IDs
jsonfile = Path('/home/mryan/Desktop/Ryan_Mike/src/butterfish/BTS_butterfish-count-ppn.json')
with jsonfile.open() as infl:
    jsondata = json.load(infl)

# as a test, work with HB200905, stn 33
# get a single file
#filename = Path('/home/mjech/Desktop/butterfish/D20090916-T135430.nc')
#filename = Path('/media/mjech/Mac Passport/HB201907/EK60_Data/netCDF4_Files/D20190727-T050056.nc')
fpath = Path(jsondata['HB200905']['33'][2])
filelist = [Path(fpath / f) for f in jsondata['HB200905']['33'][3]]
dtfmt = '%m/%d/%Y %H:%M:%S'
#begtime = pd.to_datetime(datetime.strptime(jsondata['HB200905']['33'][0], 
#                                           dtfmt)).replace(tzinfo=timezone.utc)
# begin time of the trawl - set on the bottom
begtime = pd.to_datetime(datetime.strptime(jsondata['HB200905']['33'][0], dtfmt))

edlist = []
for f in filelist:
    #edlist = [ep.open_converted(str(filename))]
    edlist = [ep.open_converted(str(f))]

#ed = ep.combine_echodata(edlist, client=client)
ed = ep.combine_echodata(edlist)

# platform and transducer information
vo=ed['Platform'].vertical_offset.isel(channel=0).values
wl=ed['Platform'].water_level.isel(channel=0).values
toz=ed['Platform'].transducer_offset_z.isel(channel=0).values
poz=ed['Platform'].position_offset_z.isel(channel=0).values

# apply calibrations and compute Sv
Sv = ep.calibrate.compute_Sv(ed)

# remove noise
# this only removes background noise
Sv = ep.preprocess.remove_noise(Sv, range_sample_num=10, ping_num=20, 
                                noise_max=-100, SNR_threshold=6)

# very quick visualization
#channeltoget = 1
#Sv0 = Sv.Sv.isel(channel=channeltoget).values
#fig = figure('Sv channel: '+str(channeltoget))
#plt.imshow(np.transpose(Sv0), cmap='viridis', vmin=-80, vmax=-30, aspect='auto', 
#           interpolation='none')
#plt.show()
#fig = figure('Echopype echogram')
#Sv.Sv.sel(frequency_nominal=120000).plot(x='ping_time', yincrease=False,
#               cmap='viridis', vmin=-80, vmax=-30)

# compute MVBS; all caps is too hard to type - use lower case
mvbs = ep.preprocess.compute_MVBS(Sv, range_meter_bin=0.5, ping_time_bin='2S')
# get list of frequencies
fqlist = list(mvbs.frequency_nominal.values)

# swap channel dimension for frequency
Sv = ep.preprocess.swap_dims_channel_frequency(Sv)
mvbs = ep.preprocess.swap_dims_channel_frequency(mvbs)

fqtoget = fqlist[3]
fig = figure('MVBS channel: '+str(int(fqtoget)))
plt.imshow(np.transpose(mvbs.Sv.sel(frequency_nominal=fqtoget).values), cmap='viridis', 
           vmin=-80, vmax=-30, aspect='auto', interpolation='none')
plt.show()

# select a single frequency and use that echogram for a number of uses
fqtoget = 120000
Sv_fq = mvbs.Sv.sel(frequency_nominal=fqtoget)

### 
# get a seabed (bottom) detection line
# use units of meters
searchmin = 10
windowlen = 11
Svbackstep = 35
#botline = afscBotDetect(tmp, search_min=searchmin, window_len=windowlen, backstep=Svbackstep)

# seabed line as a data array
botline = sbd(Sv_fq)
seabed = xr.DataArray(data=botline.afscBotDetect(search_min=searchmin, 
                                                 window_len=windowlen, 
                                                 backstep=Svbackstep),
                      dims='ping_time', 
                      coords={'ping_time':mvbs.ping_time.values})
fig = figure()
plt.plot(seabed.values)
plt.show()

###
# trawl mensuration
# height of the headrope in meters
trawl_height = 5

# we have the start time (when doors are on the bottom) and the trawl is 
# nominally 20 minutes long with the net on the bottom
trawldur = 20  # minutes

# create trawl path as dataset
# times correspond to the door set on the seabed and door off the seabed
# footrope = seabed detected line
tmp = seabed.sel(ping_time=slice(begtime, begtime+pd.to_timedelta(trawldur, unit='m')))
trawl_lines = xr.Dataset(data_vars=dict(footrope=(['ping_time'], tmp.values),
                                        headrope=(['ping_time'], tmp.values-trawl_height)),
                         coords={'ping_time':tmp.ping_time.values})

###
# select the values within the trawl path
# make a boolean mask where trawl path cells are True
btmask = Sv_fq.copy()
# use False if you want to select values above, below, or between lines
btmask.values = np.full(np.shape(btmask), False)
# use True if you want to remove values above, below, or between lines
#btmask.values = np.full(np.shape(btmask), True)

# apply to values below the footrope
#lm(btmask).apply_below_line_segment(trawl_lines.footrope, value=False)
# apply to values between the footrope and headrope
lm(btmask).apply_between_line_segments(trawl_lines.headrope, trawl_lines.footrope, 
                                       value=True)

svstack = Sv_fq.stack(z=('ping_time', 'echo_range'))
mstack = btmask.stack(z=('ping_time', 'echo_range'))
#mvbs.update(svstack.where(mstack==True, -999).unstack('z'))
btSv = svstack.where(mstack==True, -999).unstack('z')
fig = figure('Trawl MVBS channel: '+str(int(fqtoget)))
plt.imshow(np.transpose(btSv.values), aspect='auto', cmap='viridis', 
           vmin=-80, vmax=-30, interpolation='none')
plt.show()

#trawl_mvbs = mvbs.sel(ping_time=slice(begtime, begtime+pd.to_timedelta(trawldur, unit='m')), 
#                      frequency_nominal = fqtoget)
#                      echo_range = slice(50, 60))



