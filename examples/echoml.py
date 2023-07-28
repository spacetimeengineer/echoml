#!/usr/bin/python

# System tools.

import os, sys, json

# Data science tools.

import numpy as np
import pandas as pd
import xarray as xr

# Machine Learning tools.

from sklearn.cluster import KMeans
from sklearn import preprocessing
from itertools import chain, combinations

# Echosounder tools.

import echopype as ep
from echopype import open_raw
import echoregions as er

# Plotting tools.

import matplotlib.pyplot as plt
import itertools

# Logging tools.

from loguru import logger

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)

class FrequencyData():
    """Given some dataset 'Sv', list all frequencies available. This class offers methods which help map out frequencies and channels plus additional utilities. 
    """   
    
    def __init__(self, Sv):
        """Initializes class object and parses the frequencies available within the echdata object (xarray.Dataset) 'Sv'.

        Args:
            Sv (xarray.Dataset): The 'Sv' echodata object.
        """
        
        self.Sv = Sv # Crreate a self object.
        self.frequency_list = [] # Declares a frequency list to be modified.
        
        self.construct_frequency_list() # Construct the frequency list.
        #TODO : This string needs cleaning up ; remove unneeded commas and empty tuples.
        self.frequency_set_combination_list = self.construct_frequency_set_combination_list() # Constructs a list of available frequency set permutations. Example : [('18 kHz',), ('38 kHz',), ('120 kHz',), ('200 kHz',), ('18 kHz', '38 kHz'), ('18 kHz', '120 kHz'), ('18 kHz', '200 kHz'), ('38 kHz', '120 kHz'), ('38 kHz', '200 kHz'), ('120 kHz', '200 kHz'), ('18 kHz', '38 kHz', '120 kHz'), ('18 kHz', '38 kHz', '200 kHz'), ('18 kHz', '120 kHz', '200 kHz'), ('38 kHz', '120 kHz', '200 kHz'), ('18 kHz', '38 kHz', '120 kHz', '200 kHz')]
        # print(self.frequency_set_combination_list)
        self.frequency_pair_combination_list = self.construct_frequency_pair_combination_list() # Constructs a list of all possible unequal permutation pairs of frequencies. Example : [('18 kHz', '38 kHz'), ('18 kHz', '120 kHz'), ('18 kHz', '200 kHz'), ('38 kHz', '120 kHz'), ('38 kHz', '200 kHz'), ('120 kHz', '200 kHz')] 
        # print(self.frequency_pair_combination_list)
        
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
        """Constructs a list of available frequency set permutations. 
        Example : [
            ('18 kHz',), ('38 kHz',), ('120 kHz',), ('200 kHz',), ('18 kHz', '38 kHz'),
            ('18 kHz', '120 kHz'), ('18 kHz', '200 kHz'), ('38 kHz', '120 kHz'), ('38 kHz', '200 kHz'),
            ('120 kHz', '200 kHz'), ('18 kHz', '38 kHz', '120 kHz'), ('18 kHz', '38 kHz', '200 kHz'),
            ('18 kHz', '120 kHz', '200 kHz'), ('38 kHz', '120 kHz', '200 kHz'),
            ('18 kHz', '38 kHz', '120 kHz', '200 kHz')
            ]


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
    
    def __init__(self, Sv, channel_list = None, k = None, random_state = None, n_init = 10, max_iter = 300, frequency_list = None, model = "DIRECT"): # TODO Need to take in channel list instead of query matrix.
        """_summary_

        Args:
            Sv (_type_): _description_
            channel_list (_type_, optional): _description_. Defaults to None.
            k (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            frequency_list (_type_, optional): _description_. Defaults to None.
        """
        self.channel_list = channel_list # If the user chooses, they may provide a channel list insead of a specify frequency set.
        self.frequency_set_string = "" # Declare a frequency set string for simple labeling purpose with small dewscriptions of frequencies applied to kmeans.
        
        self.Sv = Sv # Echodata xarray object.
        self.frequency_list = frequency_list # Make a class object from frequency_list that was passed.
        self.simple_frequency_list = frequency_list
        self.k = k # KMeans configuration variable. The cluster count.
        self.random_state = random_state # Class variable 'random_state' is a general kmeans parameter.
        self.model = model # KMeans configuration variable. Pre-clustering DF model. Constructed from Sv.Sv and dictates which dataframe is fed into the KMeans clustering operation.
        self.n_init = n_init # KMeans configuration variable. 
        self.max_iter = max_iter # KMeans configuration variable. Max iterations.
        
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
            pd.DataFrame: The dataset which is directly fed into KMeans.
        """        
        pre_clustering_df = pd.DataFrame() # Declare empty df which will eventually conatin columns of 'Sv' value columns ripped from DataFrames which were converted from DataArrays. This is like a flattening of dimensionalities and allows 'Sv' to be represented as a single column per frequency.
        sv_frequency_map_list = [] # Declare empty list to conatin equivilent copies of the clustering data through iteration. This redundancy is tolerated becasue it allows a clean mapping give then
        sv_frequency_absolute_difference_map_list = []
        
        self.frequency_pair_combination_list = list(itertools.combinations(self.frequency_list, 2))
        
        if self.model == "DIRECT": # The DIRECT clustering model clusters direct Sv.Sv values. 
 
 
            for i in self.frequency_list: # Need a channel mapping function.
                channel_df = self.Sv.Sv[i[0]].to_dataframe(name=None, dim_order=None) # Convert Sv.Sv[channel] into a pandas dataframe.
                channel_df.rename(columns = {'Sv':str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"}, inplace = True) # Rename the column to the frequency associated with said channel. This value is pulled from the xarray.
                sv_frequency_map_list.append(channel_df[str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"]) # Append columns with each frequency or channel. (It is always best to retain a map between channels and frequencies.)
                
            pre_clustering_df = pd.concat(sv_frequency_map_list, axis = 1) # Creats a new dataframe from the values previously constructed. This is done to keep it steril.
        
        if self.model == "ABSOLUTE_DIFFERENCES": # The ABSOLUTE_DIFFERENCES clustering model clusters the absolute value of the differences between a pair permutation of frequency based Sv.Sv values. 
            # Each pair permutation is given it's own column within the dataframe that is fed into KMeans in the same way that with DIRECT each frequency is given it's own column within the dateframe that is fed into KMeans.
            # In other words this model was built to solve the problems of DIRECT by not allowing identical frequencies to be clustered together meaningfully becasue there should no no new information produced by that. 
            # If you attempt to feed the same frequencies in you will get a blank screen. This mean that 100% of the visual information is meaningful.
        
            for i in self.frequency_pair_combination_list:
                                
                sv_frequency_absolute_difference_df = (self.Sv.Sv[i[0][0]].to_dataframe(name=None, dim_order=None)["Sv"] - self.Sv.Sv[i[1][0]].to_dataframe(name=None, dim_order=None)["Sv"]).abs().values
                #print(sv_frequency_absolute_difference_df)
                index_name = "abs(Sv("+str(self.Sv.Sv[i[0][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz" +")-Sv("+ str(self.Sv.Sv[i[1][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz))"
                x = pd.DataFrame(data = sv_frequency_absolute_difference_df, columns = [index_name])
                sv_frequency_absolute_difference_map_list.append(x[index_name])
        
        
            #print(sv_frequency_absolute_difference_map_list)
            pre_clustering_df = pd.concat(sv_frequency_absolute_difference_map_list, axis = 1) # Version 1 
            
        return pre_clustering_df.reset_index().select_dtypes(include=['float64']) # Returns pre-clustering dataframe.


    def assign_sv_clusters(self):
        """_summary_
        """        
        self.pre_clustering_df = self.construct_pre_clustering_df() # Construct dataframe which is to be fed into the kmeans clustering algorithm. In this each column is a frequency.
        logger.info(self.model + " Preclustering Dataframe :")
        print("")                                   # Logging message.
        print(self.pre_clustering_df)               # Logging message.
        logger.info("Normalizing")                  # Logging message.
        df_normalized = preprocessing.scale(self.pre_clustering_df) # Normalize dataset such that values are from 0 to 1.
        logger.info("Normalized dataframe : ")      # Logging message.
        print("")                                   # Logging message.
        print(df_normalized)                        # Logging message.
        
        self.df_clustered = pd.DataFrame(df_normalized) # Make into a dataframe.
        logger.info("Calculating KMeans")
        kmeans = KMeans(n_clusters=self.k, random_state = self.random_state, init='k-means++', n_init=10, max_iter=300)  # Kmeans configuration object.
        X = self.df_clustered.values # 'X' is the sklearn convention. 
        clustered_records = kmeans.fit_predict(X) # The clustering data in df format.
        self.Sv_df = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None) # We only need to borrow the dimensionality of the xarray so we only need one element of self.Sv.Sv.
        self.Sv_df[self.frequency_set_string] = clustered_records + 1 # So adding one will keep cluster group numbers non-zero.
        
        self.clustering_data = self.Sv_df.to_xarray() # Since the dimensionality is correct at this point we can safely convert this df into an xarray and we can make modifications later.
        km_cluster_maps = [] # An array which helps prepare the dataArray dimensionality. It will store copies of clustering data. The redundancy may seem unfortune but it is infact nessecary due to the nature of how the cluster map was created. 
        
        for i in range(len(self.Sv.Sv)): # For each frequency, map an euqal clustering set to satisfy dimensionality constraints. This is nuanced but nessecary to make the DataArray meaningful and sensable.
            #TODO: I know this looks strange but it is not trust me.
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
    """_This class essentially serves as a parametrization wrapper to sklearn/KMeans configurations, 
    echopype configurations, MVBS configurations, echodata configurations, echoregion configurations, 
    matplotlib configuration and pre-clustering configurations. Currently as of 07/01/2023 this is the parent function. 
    Note : It is likely that this object class will eventually be encapsulated by more advanced function sets in the future.
    Note : This function is not yet stable becasue of MVBS considerations.
    """
    def __init__(self, file_path, save_path, sonar_model, frequency_list, cluster_count, random_state = None, n_init = 10, max_iter = 300, model = "DIRECT", color_map = "viridis", plot = True, data_reduction_type = None, range_meter_bin = None, ping_time_bin = None, range_sample_num = None, ping_num = None, remove_noise = True, ping_time_begin = None, ping_time_end = None, range_sample_begin = None, range_sample_end = None, line_files = None, region_files = None ):
        """_summary_

        Args:
            file_path (str): url or local path of .raw or .nc file containing echosounder data.
            save_path (str): The filepath to where the user desires resources generated by this class to be saved.
            frequency_list (list): Frequency list of the form : ["<int>kHz","<int>kHz","<int>kHz"] -> Example : ["38kHz","120kHz","200kHz"]
            cluster_count (int): Integer ciunt of the clusters that tha data will be broken down into.
            random_state (int, optional): _description_. Defaults to None.
            model (str, optional): _description_. Defaults to "DIRECT".
            color_map (str, optional): Colormap associated with the matplotlib figure() object. Defaults to "viridis".
            plot (bool, optional): _description_. Defaults to True.
            plot_relevent_echograms (bool, optional): _description_. Defaults to True.
            data_reduction_type (str, optional): MVBS parametrization. Defaults to None.
            range_meter_bin (_type_, optional): MVBS parametrization. Defaults to None.
            ping_time_bin (_type_, optional): MVBS parametrization. Defaults to None.
            range_sample_num (_type_, optional): MVBS parametrization. Defaults to None.
            ping_num (_type_, optional): MVBS parametrization. Defaults to None.
            remove_noise (bool, optional): Noise removal intention boolean state variable. Defaults to True.
            ping_time_begin (_type_, optional): Rectangular areas of interest parameter. ping_time begin.. Defaults to None.
            ping_time_end (_type_, optional): Rectangular areas of interest parameter. ping_time end.. Defaults to None.
            range_sample_begin (_type_, optional): Rectangular areas of interest parameter. range_sample begin.. Defaults to None.
            range_sample_end (_type_, optional): Rectangular areas of interest parameter. range_sample end.. Defaults to None.
        """        
        
        self.file_path = file_path # The .raw or .nc filepath.
        self.file_name = self.file_path.split("."+self.file_path.split(".")[-1])[0].replace(".","").split("/")[-1] # The filename without the path prefix or extension.
        self.save_path = save_path # The save path of the echograms, cluster maps and other resources.
        
        # KMeans configuration
        
        self.frequency_list = frequency_list
        print(type(self.frequency_list))
        self.cluster_count = cluster_count # The value 'k'. the integer number of clusters ( or colors ) the plot will display.
        self.random_state = random_state
        self.model = model # Preclustering model.
        self.n_init = n_init
        self.max_iter = max_iter # 
        
        self.color_map = color_map # Color profile of echogram and cluster map plots.
        self.plot = plot # Boolean state of weather or not plotting is enabled.
        self.data_reduction_type = data_reduction_type # MVBS parametrization.
        self.range_meter_bin = range_meter_bin # MVBS parametrization.
        self.ping_time_bin = ping_time_bin # MVBS parametrization.
        
        self.range_sample_num = range_sample_num # MVBS parametrization.
        self.ping_num = ping_num # MVBS parametrization.
        
        self.remove_noise = remove_noise # Noise removal intention boolean state variable.
        
        self.ping_time_begin = ping_time_begin # Rectangular areas of interest parameter. ping_time begin.
        self.ping_time_end = ping_time_end # Rectangular areas of interest parameter. ping_time end.
        self.range_sample_begin = range_sample_begin # Rectangular areas of interest parameter. range_sample begin.
        self.range_sample_end = range_sample_end # Rectangular areas of interest parameter. range_sample end.

        self.frequency_list_string = self.__construct_frequency_list_string()
        self.sonar_model = sonar_model # The Echosounder model string which Echopype requires as base parametrization when converting .raws or .nc files.
        
        self.line_files = line_files # Line files list. Files are provided from echoview.
        self.region_files = region_files # Region files list. Files are provided from echoview.
        
        
        
        self.__construct_kmeans_cluster_map() # Main execution function encapsulating nessecary variable constructions.
        

    def __construct_kmeans_cluster_map(self):
        """Once this class object is instantiated, a run function is employed to perform some required initialization sequences so that all class object variables can be constructed. 
        Another way to think about it is that run() handles the top level abstraction of KMeanClusterMap() managment and Sv prep for KMeans clustering analysis. 
        This involves converting, cropping, dropping Nans, computing MVBSusing physical units or sample number, 
        removing noise and applying .EVL or .EVR files.
        """        
        
        if self.file_path is None or ( self.file_path.split(".")[-1] != "nc" and self.file_path.split(".")[-1] != "raw" ):
            
            logger.error("Provide valid .raw or .nc file")                                                            # Logging error message.
            
        else:
            
            
            if not os.path.exists(self.save_path): # If directory does not exist.
                
                os.makedirs(self.save_path)
            

            self.Sv = self.__compute_Sv(self.file_path, self.sonar_model)    
                
            self.Sv = self.__RROI(self.Sv, self.ping_time_begin, self.ping_time_end, self.range_sample_begin, self.range_sample_end )
            
            self.Sv = self.__drop_nans(self.Sv)

            if self.remove_noise == True:
                
                self.Sv = self.__remove_noise(self.Sv)
                    
            self.Sv = self.__configure_MVBS(self.Sv)                   # Process routine. This deals with MVBS parametrization.
                
            #if self.line_files != None:
                
                #self.Sv = self.__process_line_files(self.line_files)
                
            if self.region_files != None:
                
                self.Sv = self.__process_region_files(self.region_files)

            logger.info('Preparing KMeans clustering.') # Logging message.
            self.kmeans_operation = KMeansOperator( Sv = self.Sv,  frequency_list = self.frequency_list, k = self.cluster_count, random_state = self.random_state, n_init = self.n_init, max_iter = self.max_iter,model = self.model )  
            self.frequency_map = self.kmeans_operation.frequency_list # Makes a copy of the constructed frequency list data for this class since we need it for plotting.

            if self.save_path != None: # If a save path was provided.     
                
                logger.info('Saving cluster map and corresponding echograms...')    # Logging message.
                self.__full_save(self.kmeans_operation.Sv) # This saves the kmeans cluster map and a corresponding echogram for each involved frequency.
            
            if self.plot == True:
                
                logger.info('Plotting cluster map and corresponding echograms...')    # Logging message.    
                plt.show()
            
            
    def __compute_Sv(self, file_path, sonar_model):
        
        if self.file_path.split(".")[-1] == "raw": # If sonar data file provided was a .raw file.

            logger.info('Attempting to convert (' + file_path + ") to a .nc format...")                                    # Logging message.
            
            ed = ep.open_raw(raw_file = file_path, sonar_model=sonar_model) # Principle echopype conversion function. This provides echodata object.
            logger.info('Saving .nc file converted from .raw format as (' + "/nc_files" + file_path + ")...")     # Logging message.
            
            ed.to_netcdf(save_path='./nc_files')
            Sv = ep.calibrate.compute_Sv(ed)
            logger.info("Converted!")
            
            return Sv

        if self.file_path.split(".")[-1] == "nc": # If a .nc file is provided by user.

            ed = ep.open_converted(self.file_path) # Principle echopype conversion function. This provides echodata object.
            self.Sv = ep.calibrate.compute_Sv(ed)
        
            return Sv
        
    def __process_line_files(self):
        
        if self.line_files != None:
            
            pass

    def __process_region_files(self):
        
        if self.region_files != None:
            
            # TODO : Line fiels should be read befor or after MVBS
            Sv = self.Sv
            
            # create depth coordinate:
            echo_range = self.Sv.echo_range.isel(channel=0, ping_time=0)
            # assuming water levels are same for different frequencies and location_time

            depth = self.Sv.water_level.isel(channel=0, time3=0) + echo_range
            depth = depth.drop_vars('channel')
            # creating a new depth dimension

            Sv['depth'] = depth
            Sv = Sv.swap_dims({'range_sample': 'depth'})
            logger.info("Reading region and line files.")                                                   # Logging message.
            
            
            
            
            EVR_FILE = self.region_files[0]     #TODO : Make this work first then process whole list.
            r2d = er.read_evr(EVR_FILE)
            print(r2d.data.region_id.values)
            M = r2d.mask(Sv.isel(channel=0).drop('channel'), [1], mask_var="ROI")
            logger.info("M : ")
            logger.info(M.max())
            plt.figure()
            M.plot()
            
            Sv_masked = Sv.where(M.isnull())
            plt.figure()
            Sv_masked.isel(channel=0).T.plot(yincrease=False)    
            
    def __RROI(self, Sv, ping_time_begin, ping_time_end, range_sample_begin, range_sample_end ):
        logger.info("Rectangular region of interest is cropped out of analysis region.")
        Sv_RROI = Sv.isel(range_sample=slice(range_sample_begin, range_sample_end), ping_time=slice(ping_time_begin, ping_time_end))
        logger.info("Region Cropped!")
        return Sv_RROI        
    
    def __drop_nans(self, Sv):
        
        logger.info("Dropping NaNs") 
        Sv_naNs_dropped = Sv.dropna(dim="range_sample", how="any") # Pulls a cleaner version of Sv without NaNs.
        logger.info("NaNs Dropped!")
        return Sv_naNs_dropped
    
    def __remove_noise(self, Sv):
                    
        logger.info('Removing noise from Sv...')                            # Logging message.
        logger.info('   range_sample = ' + str(self.range_sample_num))      # Logging message.
        logger.info('   ping_num = ' + str(self.ping_num))                  # Logging message.
        logger.info("Noise Removed!")

        return ep.clean.remove_noise(Sv, range_sample_num=self.range_sample_num, ping_num=self.ping_num)  
      
    def __configure_MVBS(self, Sv):
        """Configure MVBS using provided class variables. This internal method should not be directly employed by user.
        """
        if self.range_meter_bin != None and self.ping_time_bin != None:
            logger.info('Calculating MVBS using reduction by physical units.')  # Logging message.
            logger.info('   range_meter_bin = ' + str(self.range_meter_bin))    # Logging message.
            logger.info('   ping_time_bin = ' + str(self.ping_time_bin))        # Logging message.
            self.Sv = ep.commongrid.compute_MVBS(Sv, range_meter_bin = self.range_meter_bin, ping_time_bin = self.ping_time_bin )
                
        if self.ping_num != None and self.range_sample_num != None:
            logger.info('Calculating MVBS using reduction by sample number.')
            logger.info('   range_sample_num = ' + str(self.range_sample_num))
            logger.info('   ping_num = ' + str(self.ping_num))
            self.Sv = ep.commongrid.compute_MVBS_index_binning( Sv, range_sample_num=self.range_sample_num, ping_num=self.ping_num )
                
        return Sv
        
    def __save_echogram(self, data_array, channel):
        """_summary_

        Args:
            data_array (xarray.DataArray): _description_
            channel (str): Integer represented as a string.
        """        
        plt.figure() # Instantiates a matplotlib figure object which is required for plotting multiple figures. Required for plotting and saving. Recalled when plt.show() is called.
        cmap = plt.get_cmap(self.color_map, self.cluster_count) # The plots color scheme. There are various options. See above.     
        data_array[channel].transpose("range_sample","ping_time").plot(cmap = cmap)
        plt.title("frequency = "+self.__get_frequency(channel)+",    file = "+self.file_path+",    colormap = "+self.color_map ) # Plot configuration.    
        plt.gca().invert_yaxis() # Since the image flips upside down for some reason, this is applied as a correction.
        plt.savefig(fname = self.save_path+"/eg:"+self.file_name+"<"+self.__get_frequency(channel)+">", dpi=2048) # Save a parametrized filename. # File naming/saving configuration.  
    
        
    def __save_cluster_map(self, Sv):
        """Saves the Kmeans cluster map.

        Args:
            Sv (_type_): Principle xarray object 
        """        
        
        plt.figure() # Instantiates a matplotlib figure object which is required for plotting multiple figures. Required for plotting and saving. Recalled when plt.show() is called.
        cmap = plt.get_cmap(self.color_map, self.cluster_count) # The plots color scheme. There are various options. See above.
        
         
        Sv["km_cluster_map"+self.kmeans_operation.frequency_set_string][0].transpose("range_sample","ping_time").plot(cmap = cmap)
        plt.title(self.kmeans_operation.frequency_set_string+",    cluster_count = "+str(self.cluster_count)+",    random_state = "+str(self.random_state)+",    file = "+self.file_path+",    colormap = "+self.color_map) # Plot configuration.
        plt.gca().invert_yaxis() # Since the image flips upside down for some reason, this is applied as a correction.
        plt.savefig(self.save_path+"/km:"+self.file_name+"<"+ self.frequency_list_string+"k="+str(self.cluster_count)+"_rs="+str(self.random_state)+"_cm="+self.color_map+"_md="+str(self.model)+"_rmb="+str(self.range_meter_bin)+">", dpi=2048) # File naming/saving configuration.


    def __full_save(self, Sv):
        """Saves a KMeans cluster map and associated echograms to the designated directory defined by save_path parameter.

        Args:
            Sv ((Modified Echodata Object) xarray.DataArray): Echodata object. Principle object of echopype library.
        """        
        self.__save_cluster_map(Sv) # Save the kmeans cluster Map image.
        for frequency in self.frequency_map: # For each frequency record within the frequency list. Example : [1, '38 kHz'], [2, '120 kHz'], [0, '18 kHz'], [3, '200 kHz']
            self.__save_echogram(self.Sv["Sv"],frequency[0]) # Save the echogram image.

    def __construct_frequency_list_string(self):
        """Returns a string which serves as a representation of which frequencies are utilizednin KMeans. Example : "38kHz_70kHz_120kHz_200kHz_". For use in file and variable naming mostly.

        Returns:
            str: Example : "38kHz_70kHz_120kHz_200kHz_"
        """        
        frequency_list_string = ""
        for frequency in self.frequency_list:
            frequency_list_string = frequency_list_string + frequency+"_"
        return frequency_list_string 

    def __get_frequency(self, channel):
        """For a given integer channel returns a string frquency of the form : "<int>kHz" ; Example : "200kHz" ,  associated with class variable 'self.Sv.Sv[channel]'

        Args:
            channel (int): Integer channel which maps to desired frequency.

        Returns:
            str: Returns a string frquency of the form : "<int>kHz" ; Example : "200kHz" ,  associated with class variable 'self.Sv.Sv[channel]'
        """        
        for frequency in self.frequency_map: # For each frerquency in the frequency map.
            if frequency[0] == channel: # If given and recorded channels match.
                return frequency[1] # Extract corresponding frequency of matched channel from frequency list ( or frequency map I should call it.)

        
            
    def __get_channel(self, frequency):
        """For a given frequency of the form : "<int>kHz" ; Example : "200kHz" , returns the integer channel as a string associated with class variable 'self.Sv'

        Args:
            frequency ("<int>kHz"): For a given frequency of the form : "<int>kHz" ; Example : "200kHz". Matching frequencies only.

        Returns:
            str: The channel an integer represented as a string. 
            Example : "3"
        """        
        for frequency in self.frequency_map:   # For each frerquency in the frequency list.
            if frequency[1] == frequency:       # If given and recorded frequencies match.
                return frequency[0]             # Extract corresponding channel of matched frequency from frequency list ( or frequency map I should call it.)


def main(): # Defines main method. This source code serves as a python module as well as an executable script becasue it can be run if a json is fed in. For example ; $ python3 echoml.py test.json 
    """Defines main method. This method makes this source code executable if json is passed. An example call ; $ python3 echoml.py km_config.json
    """
    for i in range(len(sys.argv)): # For each json argument passed to python file.
        
        if i == 0: # Ignore first argument ('python or python3').
            
            pass # Pass.
        
        else: # For every other argument.
            
            # Open JSON file
            logger.info('Opening '+sys.argv[i])
            
            f = open(sys.argv[i]) # File object which can be used to create python class object representations of files.
            data = json.load(f) # Create json data object compadible with python functionality.
            json_formatted_str = json.dumps(data, indent=4) # Logs the config json.
            logger.info(json_formatted_str) # Logging message.
            KMClusterMap(
        
                # Path Configuration
                EK_data_path =  data["path_config"]["EK_data_path"],
                EK_data_filenames =  data["path_config"]["EK_data_filenames"],
                save_path = data["path_config"]["save_path"],                                       # Save directory where contents will be stored. 
                
                # Path Configuration
                
                sonar_model = data["sonar_model"], 
                
                # KMeans Configuration
                
                frequency_list = data["kmeans_config"]["frequency_list"],                           # List of frequencies to be included into the KMeans clustering operation. There must be atleast two otherwise operation is meaningless.
                cluster_count = data["kmeans_config"]["cluster_count"],                             # The quantity of clusters (colors) the data is clustered into.
                random_state = data["kmeans_config"]["random_state"],                               # Random state variable needed for KMeans operations.
                
                n_init = data["kmeans_config"]["n_init"],
                max_iter = data["kmeans_config"]["max_iter"],
                
                model = data["kmeans_config"]["model"],                                             # Paramaters may be "DIRECT" or "ABSOLUTE_DIFFERENCES" and defaults to "DIRECT". This reffers to the way the Sv values of a given set of frequencies are being compared in the clustering algorithm.
                
                
                # Plotting
                
                color_map = data["plotting"]["color_map"],                                          # Color map variable. Defaults to "viridis" Check out https://matplotlib.org/stable/tutorials/colors/colormaps.html to see options. Examples include 'jet', 'plasma', 'inferno', 'magma', 'cividis'.
                plot = data["plotting"]["plot"],                                                    # Plot the cluster map. ( As opposed to just saving the map. )
                    
                
                # MVBS & Data Reduction
                
                data_reduction_type = data["data_reduction"]["data_reduction_type"],                # Must be one of two string options, "physical_units" or "sample_number" or comment out to default to None .
                range_meter_bin = data["data_reduction"]["range_meter_bin"],                        # Range meter resolution .
                ping_time_bin = data["data_reduction"]["ping_time_bin"],                            # Ping time resolution .
                range_sample_num = data["data_reduction"]["range_sample_num"],                      # Range sample resolution.
                ping_num = data["data_reduction"]["ping_num"],                                      # Ping sample resoluition.
                
                    
                # Noise Removal
                
                remove_noise = data["noise_removal"],                                               # Removes noise.
                
                
                # Rectangular Subselection
                
                ping_time_begin = data["sub_selection"]["rectangular"]["ping_time_begin"],          # For rectangular datasubset selection. Select integer or datetime value.
                ping_time_end = data["sub_selection"]["rectangular"]["ping_time_end"],              # For rectangular datasubset selection. Select integer or datetime value.
                range_sample_begin = data["sub_selection"]["rectangular"]["range_sample_begin"],    # For rectangular datasubset selection. Select integer value.
                range_sample_end = data["sub_selection"]["rectangular"]["range_sample_end"],        # For rectangular datasubset selection. Select integer value.
                
                
                # Line Files       
                
                line_files = data["sub_selection"]["line_files"],                                   # Line files.
                region_files = data["sub_selection"]["region_files"]                                # Region files.
                
            )
            
            f.close() # Close json configuration file.
            
            

if __name__=="__main__": # If script is executed as an argument to 'python'.
    
    main() # Run main method.
