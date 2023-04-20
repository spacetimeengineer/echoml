#!/usr/bin/env python3


import echopype as ep
from echopype import open_raw
from pathlib import Path
import json





ed = open_raw('butterfish/D20090916-T132105.raw', sonar_model='EK60')  # for EK80 file
ed.to_netcdf(save_path='./unpacked_files')


'''
Do we take the difference in parrelell values between channels or due we plug the values directly into the K-Means Agent? I Suspect the 
latter is equivilent or I might say that KMeans I think already utilizes a kind of multidimensional pythagagorian or euclidian operations 
for organizing clusters.
'''

'''
class Sv_ml(self, Sv):
  
  def self.restrict_channel(self, *args):
      self.restricted_channels = *args

      self.Sv.remove()

  def self.render_ml_Sv(self, Sv):

      selfml_Sv=[] 

      for i in Sv.Sv: # Run through every channel.
          for j in Sv.Sv[i]: # Run through every sample.
              for k in Sv.Sv[i][j]: # Run though every

                   self.ml_Sv=[].append[i][j][k] # Sort of like a machine learing equivilent to Sv. It is created here. THe values are synthetica and this object can be plugged into kmeans.

      return ml_Sv # Return the new Xarray.

'''
      
      
  
