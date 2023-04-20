import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
import echopype as ep
from echopype import open_raw
from sklearn import datasets, preprocessing
import matplotlib.pyplot as plt
from itertools import chain, combinations

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)


class RandomForestAgent():
    
    def __init__(self, df, channel_list, k, random_state): # TODO Need to take in channel list instead of query matrix.

        self.random_state = random_state
        self.channel_list = channel_list
        self.k = k
        self.df = df
        
        self.assign_random_forests() # Initiate Random-Forest sequence.
        
    
    def assign_random_forests(self):
        pass
    

class Frequencies():
    """Given some dtaset 'Sv', list all frequencies available.
    """   
    
    def __init__(self, Sv):
        """Initializes class object and parses the frequencies available in the xarray 'Sv'

        Args:
            Sv (_type_): The 'Sv' echodata object.
        """
        
        self.Sv = Sv # Crreate a self object.
        self.frequency_list = [] # Declares a frequency list to be modified.
        self.construct_frequency_list() # Construct the frequency list.
        self.frequency_combination_list = self.generate_frequency_set_combination_list()


    def construct_frequency_list(self):
        """Parses the frequencies available in the xarray 'Sv'
        """
        for i in range(len(self.Sv.Sv)): # Iterate through the natural index associated with Sv.Sv .
            
            self.frequency_list.append(str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz") # Extract frequency.
            
        return self.frequency_list # Return string array frequency list of the form [18kHz, 70kHz, 200 kHz]
        
        
    def powerset(self, iterable):
        """Generates combinations of elements of iterables ; powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

        Args:
            iterable (_type_): A list.

        Returns:
            _type_: Returns combinations of elements of iterables.
        """        
        
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


    def generate_frequency_set_combination_list(self):
        
        return list(self.powerset(self.frequency_list))
    
    
    def print_frequency_set_combination_list(self):
        
        for i in self.frequency_combination_list:
            
            print(i)
            
            
    def print_frequency_list(self):
        
        for i in self.frequency_list:
            
            print(i)


class KMeansOperator: # Reference: https://medium.datadriveninvestor.com/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c
    
    def __init__(self, Sv, channel_list = None, k = None, random_state = None, frequency_list = None): # TODO Need to take in channel list instead of query matrix.

        self.random_state = random_state # Class variable 'random_state' is a general kmeans parameter.
        self.channel_list = channel_list # If the user chooses, they may provide a channel list insead of a specify frequency set.
        self.frequency_set_string = ""
        self.k = k
        self.Sv = Sv
        self.frequency_list = frequency_list
        
        if self.frequency_list == None:
            
            if self.channel_list != None:
                            
                self.frequency_list = []
                self.construct_frequency_list(frequencies_provided = False)
                self.construct_frequency_set_string()
                self.assign_sv_clusters() # Execute K-Means algorithm.
                
            else: 
                
                print("Provide a frequency_list or channel_list input parameter.")

        else:    
            self.construct_frequency_list(frequencies_provided = True)
            self.construct_frequency_set_string()
            self.assign_sv_clusters() # Execute K-Means algorithm.

        
    def construct_frequency_list(self, frequencies_provided):
        
        if frequencies_provided == True:
            self.simple_frequency_list = self.frequency_list
            self.frequency_list = []
            
            for j in self.simple_frequency_list:
                for i in range(len(self.Sv.Sv)):
                    if str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip() == j.split("kHz")[0].strip():
                        self.frequency_list.append([i,str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"])
        
        else:
            
            for i in self.channel_list:
                
                self.frequency_list.append([i,str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"])
                
            
    def construct_frequency_set_string(self):
        
        frequency_set_string = ""
        
        for i in self.frequency_list:
            
            frequency_set_string += i[1].split(" ")[0]+"kHz,"

        frequency_set_string = "<("+frequency_set_string+","
        frequency_set_string = frequency_set_string.split(",,")[0]+")>"
        self.frequency_set_string = frequency_set_string
        
        
    def construct_pre_clustering_df(self):
        
        pre_clustering_df = pd.DataFrame()
        sv_frequency_map_list = []
 
 
        for i in self.frequency_list: # Need a channel mapping function.
            
            print(i)
            channel_df = self.Sv.Sv[i[0]].to_dataframe(name=None, dim_order=None)
            channel_df.rename(columns = {'Sv':str(self.Sv.Sv[i[0]].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"}, inplace = True)
            sv_frequency_map_list.append(channel_df[str(self.Sv.Sv[i[0]].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"])
            
        pre_clustering_df = pd.concat(sv_frequency_map_list, axis = 1)
        return pre_clustering_df.reset_index().select_dtypes(include=['float64'])


    def assign_sv_clusters(self):
        
        self.pre_clustering_df = self.construct_pre_clustering_df()
        
        df_normalized = preprocessing.scale(self.pre_clustering_df) # Normalize dataset such that values are from 0 to 1.
        self.df_clustered = pd.DataFrame(df_normalized) # Make into a dataframe. TODO : Investigate and write better comments.
        kmeans = KMeans(n_clusters=self.k, random_state = self.random_state, init='k-means++', n_init=10, max_iter=300) 
        
        X = self.df_clustered.values # 'X' is the sklearn convention. 
        
        clustered_records = kmeans.fit_predict(X) # The clustering data in df format.

        self.Sv_df = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None) # We only need to borrow the dimensionality of the xarray so we only need one element of self.Sv.Sv.
        self.Sv_df[self.frequency_set_string] = clustered_records + 1 
        
        self.clustering_data = self.Sv_df.to_xarray()
        km_cluster_maps = []
        for i in range(len(Sv.Sv)):
            km_cluster_maps.append(self.clustering_data[self.frequency_set_string].values)
        
        print("km_cluster_map"+self.frequency_set_string)
        self.Sv["km_cluster_map"+self.frequency_set_string] = xr.DataArray(
            data = km_cluster_maps, dims = ['channel','ping_time','range_sample'],
            attrs = dict(
                description = "The kmeans cluster group number for a given dataset (Sv.Sv), a frequency clustering set and a total cluster count 'k'.",
                units = "Unitless",
                clusters = self.k,
                km_frequencies = self.frequency_set_string,
                random_state = self.random_state,
            ))
    





echodata = ep.open_converted("D20090916-T132105.nc")                                            # Create an EchoData object from converted .nc file.
Sv = ep.calibrate.compute_Sv(echodata).dropna(dim="range_sample", how="any")                    # Obtain a xarray object containing Sv, echo_range.


print(Sv)


frequencies = Frequencies(Sv)
frequencies.print_frequency_list()
frequencies.print_frequency_set_combination_list()

kmeans_agent = KMeansOperator(Sv, channel_list = [1,2,3], k = 10, random_state = 42)
kmeans_agent = KMeansOperator(Sv, frequency_list = ['38kHz', '120kHz', '200kHz'], k = 10, random_state = 22)    # Kmeans agent.


print(Sv)

print(Sv["km_cluster_map<(38kHz,70kHz,120kHz)>"])


