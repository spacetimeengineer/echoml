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
from itertools import chain, combinations
import matplotlib.pyplot as plt
import itertools
import os

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)


class RandomForestAgent():
    """_summary_
    """    
    def __init__(self, df, channel_list, k, random_state): # TODO Need to take in channel list instead of query matrix.
        """_summary_

        Args:
            df (_type_): _description_
            channel_list (_type_): _description_
            k (_type_): _description_
            random_state (_type_): _description_
        """
        self.random_state = random_state
        self.channel_list = channel_list
        self.k = k
        self.df = df
        
        self.assign_random_forests() # Initiate Random-Forest sequence.
        
    
    def assign_random_forests(self):
        pass
    

class FrequencyData():
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
        self.frequency_set_combination_list = self.construct_frequency_set_combination_list()
        self.frequency_pair_combination_list = self.construct_frequency_pair_combination_list()


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
        
        s = list(iterable) # Make a list from the iterable.
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) # Returns a list of tuple elements containing combinations of elements which derived from the iterable object.


    def construct_frequency_set_combination_list(self):
        """Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.

        Returns:
            list<tuple>: A list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
        """  
              
        return list(self.powerset(self.frequency_list)) # Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
    
    
    def print_frequency_set_combination_list(self):
        """Prints frequency combination list one element at a time.
        """
                
        for i in self.frequency_set_combination_list:  # For each frequency combination associated with Sv.
            print(i) # Print out frequency combination tuple.
            
            
    def construct_frequency_pair_combination_list(self):
        """Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.

        Returns:
            list<tuple>: A list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
        """  
              
        return list(itertools.combinations(self.frequency_list, 2)) # Returns a list of tuple elements containing frequency combinations which is useful for the KMeansOperator class.
    
    def print_frequency_pair_combination_list(self):
        """Prints frequency combination list one element at a time.
        """
                
        for i in self.frequency_pair_combination_list:  # For each frequency combination associated with Sv.
            print(i) # Print out frequency combination tuple.
            

            
    def print_frequency_list(self):
        """Prints each frequency element available in Sv.
        """        
        
        for i in self.frequency_list:# For each frequency in the frequency_list associated with Sv.
            print(i) # Print out the associated frequency.


class KMeansOperator: # Reference: https://medium.datadriveninvestor.com/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c
    
    def __init__(self, Sv, channel_list = None, k = None, random_state = None, frequency_list = None, model = "absolute_differences"): # TODO Need to take in channel list instead of query matrix.
        """_summary_

        Args:
            Sv (_type_): _description_
            channel_list (_type_, optional): _description_. Defaults to None.
            k (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            frequency_list (_type_, optional): _description_. Defaults to None.
        """
        self.random_state = random_state # Class variable 'random_state' is a general kmeans parameter.
        self.channel_list = channel_list # If the user chooses, they may provide a channel list insead of a specify frequency set.
        self.frequency_set_string = "" # Declare a frequency set string for simple labeling purpose with small dewscriptions of frequencies applied to kmeans.
        self.k = k # Cluster count.
        self.Sv = Sv # Echodata xarray object.
        self.frequency_list = frequency_list # Make a class object from frequency_list that was passed.
        self.simple_frequency_list = frequency_list
        self.model = model
        print(self.Sv)
        if self.frequency_list == None: # If a frequency_list wasn't passed.
            
            if self.channel_list != None: # If a channel_list wasn't passed.
                            
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
        """Either using a channel_list or a frequency_list this function provides one which satisfies all requirements of this class structure. In particular the channels and frequencies involved have to be known and mapped to oneanother.

        Args:
            frequencies_provided (boolean): was a frequency_list provided at object creation? If so then 'True' if a channel_list instead was used then 'False'.
        """        
        if frequencies_provided == True:
            
            self.simple_frequency_list = self.frequency_list
            self.frequency_list = [] # Declare a frequency list to be populated with string frequencies of the form [[1,'38kHz'],[2,'120kHz'],[4,'200kHz']] where the first element is meant to be the channel representing the frequency. This is an internal object. Do not interfere.
    
            for j in self.simple_frequency_list: # For each frequency 'j'.
                
                for i in range(len(self.Sv.Sv)): # Check each channel 'i'.
                    
                    if str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip() == j.split("kHz")[0].strip(): # To see if the channel associates with the frequency 'j' .
                        
                        self.frequency_list.append([i,str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"]) # If so append it and the channel to the 'frequency_list'.
                        
        else:
            
            for i in self.channel_list:
                
                self.frequency_list.append([i,str(self.Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"])
                
            
    def construct_frequency_set_string(self):
        """So the idea behind a frequency_set_string is to serve as a quick and simple representative string which describes the kmeans clustering map and the frequency data that was employed to create it.
        """        
        frequency_set_string = "" # Declare frequency_set_string. We call it a set even though it is a list is becasue it is not meant to change but the list aspect was useful because it allowed mutability.
        
        for i in self.frequency_list: # For each frequency in the frequency_list.
            
            frequency_set_string += i[1].split(" ")[0]+"kHz,"

        frequency_set_string = "<("+frequency_set_string+"," # Start defining the frequency_set_string.
        frequency_set_string = frequency_set_string.split(",,")[0]+")>" # Finishing defining the frequency_set_string.
        self.frequency_set_string = frequency_set_string # Make 'frequency_set_string' a class object 'self.frequency_set_string' .
        
        
    def construct_pre_clustering_df(self):
        """This is the dataframe which is passed to KMeans algorithm and is operated on. This df is synthesized by taking the Sv(s) associated with various frequencies.

        Returns:
            _type_: _description_
        """        
        pre_clustering_df = pd.DataFrame() # Declare empty df which will eventually conatin columns of 'Sv' value columns ripped from DataFrames which were converted from DataArrays. This is like a flattening of dimensionalities and allows 'Sv' to be represented as a single column per frequency.
        sv_frequency_map_list = [] # Declare empty list to conatin equivilent copies of the clustering data through iteration. This redundancy is tolerated becasue it allows a clean mapping give then
        sv_frequency_absolute_difference_map_list = []
        
        self.frequency_pair_combination_list = list(itertools.combinations(self.frequency_list, 2))
        
        if self.model == "direct":
 
 
            for i in self.frequency_list: # Need a channel mapping function.
                
                #print(i) # For testing. Each element is an ordered list pair with a channel in the first element and a frequency in the second.
                channel_df = self.Sv.Sv[i[0]].to_dataframe(name=None, dim_order=None)
                channel_df.rename(columns = {'Sv':str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"}, inplace = True)
                #print(channel_df)
                sv_frequency_map_list.append(channel_df[str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"])
                
                
            #print(sv_frequency_map_list)    
            pre_clustering_df = pd.concat(sv_frequency_map_list, axis = 1) # Version 1 
            
            
            
            #print(self.frequency_pair_combination_list)
            #Make a list of all possible unordered pairs given a list of elements
            
        
        if self.model == "absolute_differences":
        
            for i in self.frequency_pair_combination_list:
                #print(self.Sv.Sv[i[0][0]].to_dataframe(name=None, dim_order=None)["Sv"] - self.Sv.Sv[i[1][0]].to_dataframe(name=None, dim_order=None)["Sv"])
                
                sv_frequency_absolute_difference_df = (self.Sv.Sv[i[0][0]].to_dataframe(name=None, dim_order=None)["Sv"] - self.Sv.Sv[i[1][0]].to_dataframe(name=None, dim_order=None)["Sv"]).abs().values
                index_name = "abs(Sv("+str(self.Sv.Sv[i[0][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz" +")-Sv("+ str(self.Sv.Sv[i[1][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz))"
                x = pd.DataFrame(data = sv_frequency_absolute_difference_df, columns = [index_name])
                

                sv_frequency_absolute_difference_map_list.append(x[index_name])
        
        
            #print(sv_frequency_absolute_difference_map_list)
            pre_clustering_df = pd.concat(sv_frequency_absolute_difference_map_list, axis = 1) # Version 1 
        
        
        return pre_clustering_df.reset_index().select_dtypes(include=['float64'])


    def assign_sv_clusters(self):
        """_summary_
        """        
        self.pre_clustering_df = self.construct_pre_clustering_df() # Construct dataframe which is to be fed into the kmeans clustering algorithm. In this each column is a frequency.
        
        print(self.pre_clustering_df)
        
        df_normalized = preprocessing.scale(self.pre_clustering_df) # Normalize dataset such that values are from 0 to 1.
        
        print(df_normalized)
        
        self.df_clustered = pd.DataFrame(df_normalized) # Make into a dataframe.
        kmeans = KMeans(n_clusters=self.k, random_state = self.random_state, init='k-means++', n_init=10, max_iter=300)  # Kmeans configuration object.
        
        X = self.df_clustered.values # 'X' is the sklearn convention. 
        
        clustered_records = kmeans.fit_predict(X) # The clustering data in df format.

        print(clustered_records)

        self.Sv_df = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None) # We only need to borrow the dimensionality of the xarray so we only need one element of self.Sv.Sv.
        self.Sv_df[self.frequency_set_string] = clustered_records + 1 # So adding one will keep cluster group numbers non-zero.
        
        self.clustering_data = self.Sv_df.to_xarray() # Since the dimensionality is correct at this point we can safely convert this df into an xarray and we can make modifications later.
        km_cluster_maps = [] # An array which helps prepare the dataArray dimensionality. It will store copies of clustering data. The redundancy may seem unfortune but it is infact nessecary due to the nature of how the cluster map was created. 
        for i in range(len(self.Sv.Sv)): # For each frequency, map an euqal clustering set to satisfy dimensionality constraints. This is nuanced but nessecary to make the DataArray meaningful and sensable.
            km_cluster_maps.append(self.clustering_data[self.frequency_set_string].values) # Multiply this entry for each channel becasue it needs to be mapped meaningfully to match the dimensionality of range_sample and ping_time.
        
        self.Sv["km_cluster_map"+self.frequency_set_string] = xr.DataArray(
            data = km_cluster_maps, dims = ['channel','ping_time','range_sample'], # Clustering data is appended with respect to ping_time and range_sample.
            attrs = dict( # Atrrributes as a dictionary.
                description = "The kmeans cluster group number.", # Assigns a description attribute to the DataArrray.
                units = "Unitless", # Assigns a physical unit attribute to the DataArray.
                clusters = self.k, # Assigns a cluster count variable 'k' attribute to the DataArray.
                km_frequencies = self.frequency_set_string, # Assigns a frequency set utilized for kmeans clustering attribute to the DataArray.
                random_state = self.random_state, # Assigns a random state attribute to the DataArray.
            ))
    



class KMClusterMap:
    """_summary_
    """
    def __init__(self, filepath, frequency_list, cluster_count, save_path = False, color_map = "viridis", random_state = None, model = "absolute_differences", plot = True, range_meter_bin = None, ping_time_bin = None, range_bin_num = None, ping_num = None, remove_noise = False):
        
        
        
        
        self.filepath = filepath
        self.file_name = self.filepath.split("."+self.filepath.split(".")[-1])[0].replace(".","").replace("/","")
        print(self.file_name)
        self.random_state = random_state
        self.color_map = color_map
        self.save_path = save_path
        self.cluster_count = cluster_count
        self.frequency_list=frequency_list
        self.model = model
        self.plot = plot
        
        self.range_meter_bin = range_meter_bin
        self.ping_time_bin = ping_time_bin
        
        self.range_bin_num = range_bin_num
        self.ping_num = ping_num
        
        self.remove_noise = remove_noise
        
        self.frequency_list_string = self.construct_frequency_list_string()
        self.run()
        
            
    def run(self):
        
        
        
        
        
        
        save_directory = self.save_path
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        
        if self.filepath.split(".")[-1] == "raw":
            
            raw_file = self.filepath
            ed = ep.open_raw(raw_file = raw_file, sonar_model='EK60')  
            
            ed.to_netcdf(save_path='./unpacked_files')
            self.Sv = ep.calibrate.compute_Sv(ed).dropna(dim="range_sample", how="any")                    
            

            
        if self.filepath.split(".")[-1] == "nc":
            
            
            
            nc_file = self.filepath
            ed = ep.open_converted(nc_file)  
            self.Sv = ep.calibrate.compute_Sv(ed).dropna(dim="range_sample", how="any")      
            # manipulate Sv              
            
            
        if self.Sv is None:
            
            print("provide .raw or .nc file")
            
        else:
            
            self.preprocess()
            
            
            # Plot kmeans cluster map
            
            self.kmeans_operation = KMeansOperator(Sv = self.Sv, frequency_list = self.frequency_list, k = self.cluster_count, random_state = self.random_state, model = self.model) 

            if self.plot == True or self.save_path == False:
                
                self.plot_cluster_map()
            
            
            if self.save_path != False:
                
                self.save_cluster_map()
            
            
            
            
    def preprocess(self):
        
        
        if self.range_meter_bin != None and self.ping_time_bin != None:
        
            # Reduce data based on physical units
            self.MVBS = ep.preprocess.compute_MVBS(
                self.Sv,               # calibrated Sv dataset
                range_meter_bin=self.range_meter_bin,  # bin size to average along range in meters
                ping_time_bin=self.ping_time_bin  # bin size to average along ping_time in seconds exakple: '20S' ?
            )

        """if self.range_bin_num != None and self.ping_num != None:


            if self.MVBS != None:
                # Reduce data based on sample number
                self.MVBS = ep.preprocess.compute_MVBS_index_binning(
                    self.MVBS,             # calibrated Sv dataset
                    range_bin_num=self.range_bin_num,  # number of sample bins to average along the range_bin dimensionm
                    ping_num=self.ping_num         # number of pings to average
                )
                
            else:
                # Reduce data based on sample number
                self.MVBS = ep.preprocess.compute_MVBS_index_binning(
                    self.Sv,             # calibrated Sv dataset
                    range_bin_num=self.range_bin_num,  # number of sample bins to average along the range_bin dimensionm
                    ping_num=self.ping_num         # number of pings to average
                )
                

            if self.remove_noise == True:
                

                if self.MVBS != None:
                    
                    self.MVBS = ep.preprocess.remove_noise(    # obtain a denoised Sv dataset
                        self.MVBS,             # calibrated Sv d  ataset
                        range_bin_num=self.range_bin_num,   # number of samples along the range_bin dimension for estimating noise
                        ping_num=self.ping_num,        # number of pings for estimating noise
                    )
                    
                else:

                    self.MVBS = ep.preprocess.remove_noise(    # obtain a denoised Sv dataset
                        self.Sv,             # calibrated Sv dataset
                        range_bin_num=self.range_bin_num,   # number of samples along the range_bin dimension for estimating noise
                        ping_num=self.ping_num,        # number of pings for estimating noise
                    )"""
        try:
            if self.MVBS != None:
                self.Sv = self.MVBS    
        except AttributeError:
            pass
              
                

    
    def plot_cluster_map(self):
        
        
        cmap = plt.get_cmap(self.color_map, self.cluster_count)
        self.Sv["km_cluster_map"+self.kmeans_operation.frequency_set_string][0].transpose("range_sample","ping_time").plot(cmap=cmap)
        
        print(self.Sv)

        plt.title(self.kmeans_operation.frequency_set_string+",    cluster_count = "+str(self.cluster_count)+",    random_state = "+str(self.random_state)+",    file = "+self.filepath+",    colormap = "+self.color_map)
        plt.gca().invert_yaxis()
        plt.show()
        
    def save_cluster_map(self):
        
        
        cmap = plt.get_cmap(self.color_map, self.cluster_count)
        self.Sv["km_cluster_map"+self.kmeans_operation.frequency_set_string][0].transpose("range_sample","ping_time").plot(cmap=cmap)
        
        print(self.Sv)

        plt.title(self.kmeans_operation.frequency_set_string+",    cluster_count = "+str(self.cluster_count)+",    random_state = "+str(self.random_state)+",    file = "+self.filepath+",    colormap = "+self.color_map)
        plt.gca().invert_yaxis()
        plt.savefig(self.save_path+"/km:"+self.file_name+"<"+ self.frequency_list_string+"k="+str(self.cluster_count)+"_rs="+str(self.random_state)+"_cm="+self.color_map+"_md="+str(self.model)+"_rmb="+str(self.range_meter_bin)+">")

    def construct_frequency_list_string(self):
        frequency_list_string = ""
        for frequency in self.frequency_list:
            frequency_list_string = frequency_list_string + frequency+"_"
        
        return frequency_list_string


class KMClusterSweep: 
    
    def __init__(self, filepath, frequency_list, total_cluster_count, save_path = ".\km_sweep_maps", random_state = None):
        
        self.filepath = filepath
    
        for i in range(total_cluster_count):
            KMClusterMap(filepath = filepath, frequency_list = ['18kHz','200kHz'], cluster_count = i, save_path = ".\km_cluster_maps", color_map="rainbow", plot = False)
            


"""import echoclassify as ec

foo = ec.KMClusterMap(
    filepath = "./D20090405-T114914.raw",
    frequency_list = ['38kHz','70kHz','120kHz','18kHz','200kHz'],
    cluster_count = 24,
    save_path = "km_cluster_maps",
    random_state = 42,
    color_map = "jet",
    range_meter_bin = 2,
    ping_time_bin = '4S',
    range_bin_num = 300,      # Ineffectual
    ping_num = 2,             # Ineffectual
    model = "direct",
    plot = True,
    remove_noise = True       # Ineffectual
    )

foo.run()"""



KMClusterMap(
    filepath = "./D20090405-T114914.raw",
    frequency_list = ['38kHz','70kHz','120kHz','18kHz','200kHz'],
    cluster_count = 24,
    save_path = "km_cluster_maps",
    random_state = 42,
    color_map = "jet",
    range_meter_bin = 2,
    ping_time_bin = '4S',
    range_bin_num = 300,      # Ineffectual
    ping_num = 2,             # Ineffectual
    model = "direct",
    plot = True,
    remove_noise = True       # Ineffectual
    )
