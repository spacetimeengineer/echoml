# -*- coding: utf-8 -*-
"""
Convert_EK60-to-netcdf

Use echopype to convert Simrad EK60 raw files to netCDF

jech
"""

import echopype as ep
from echopype import open_raw
from pathlib import Path
import json

# retrieve a single file
#filename = Path('/home/mjech/Desktop/butterfish/D20090916-T135430.raw')
#DataDirectory = filename.parent

# retrieve multiple files
#dataDirectory = Path('/home/mjech/Desktop/butterfish')
#dataDirectory = Path('/media/mjech/Mac Passport/HB200901')
#outdir = dataDirectory / 'netCDF4_Files'
#filelist = []
#for f in dataDirectory.glob('**/*.raw'):
#    print('.raw file: ', f)
#    filelist.append(f)
    
# directory paths file lists in a JSON file
# open the JSON file
json_path = Path('/media/mjech/Mac Passport/HB201907/Trawl_Data/MWT_measurements')
json_fn = json_path / 'HB201907_MWT-files.json'
#json_path = Path('/media/mjech/Mac Passport/HB202205/Trawl_Data/MWT_measurements')
#json_fn = json_path / 'HB202205_MWT-files.json'
with open (json_fn) as jsonfile:
    fnames = json.load(jsonfile)
cruiseID = fnames['cruiseID']
netID = fnames['netID']
EKmodel = fnames['EKmodel']

for towID in fnames['tow'].keys():
    mwtID = '-'.join([netID, towID])
    print('Doing: ', mwtID)

    dataDirectory = Path(fnames['EKpath'])
    outdir = dataDirectory / 'netCDF4_Files'
    EKfiles = fnames['tow'][towID]['EKfiles']
    filelist = []
    for ifl in EKfiles:
        filelist.append(str(dataDirectory / ifl))

    # convert each to netCDF4
    # create the output directory
    if (outdir.exists()):
        print('Directory %s exists' % outdir)
    else:
        try:
            outdir.mkdir()
        except OSError:
            print('Unable to create output directory %s' % outdir)
            exit()
        else:
            print('Output directory created %s' % outdir)
    
    for f in filelist:
        ed = open_raw(str(f), sonar_model=EKmodel)
        # Henry B. Bigelow ICES code is 33HH
        ed['Platform']['platform_name'] = 'Henry B. Bigelow'
        ed['Platform']['platform_type'] = 'SHIPC'
        ed['Platform']['platform_code_ICES'] = '33HH'
        # the to_netcdf function seems to have an error catch, so I don't use "try"
        ed.to_netcdf(save_path=str(outdir))
    

