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
























class KMeansOperator: # Reference: https://medium.datadriveninvestor.com/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c
    
    def __init__(self, Sv, M = None, cluster_count = None, random_state = None, n_init = 10, max_iter = 300, frequency_list = None, model = "DIRECT"): # TODO Need to take in channel list instead of query matrix.
        """_summary_

        Args:
            Sv (_type_): _description_
            channel_list (_type_, optional): _description_. Defaults to None.
            k (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            frequency_list (_type_, optional): _description_. Defaults to None.
        """
        self.frequency_set_string = "" # Declare a frequency set string for simple labeling purpose with small dewscriptions of frequencies applied to kmeans.
        
        self.Sv = Sv # Echodata xarray object.
        self.M = M
        self.frequency_list = frequency_list # Make a class object from frequency_list that was passed.
        self.simple_frequency_list = frequency_list
        self.cluster_count = cluster_count # KMeans configuration variable. The cluster count.
        self.random_state = random_state # Class variable 'random_state' is a general kmeans parameter.
        self.model = model # KMeans configuration variable. Pre-clustering DF model. Constructed from Sv.Sv and dictates which dataframe is fed into the KMeans clustering operation.
        self.n_init = n_init # KMeans configuration variable. 
        self.max_iter = max_iter # KMeans configuration variable. Max iterations.
        
        if self.frequency_list != None: # If a frequency_list wasn't passed.
             
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
        
        
        
        
        
        
        self.frequency_pair_combination_list = list(itertools.combinations(self.frequency_list, 2))
        
        if self.model == "DIRECT": # The DIRECT clustering model clusters direct Sv.Sv values. 
 
            sv_direct_list = [] # Declare empty list to conatin equivilent copies of the clustering data through iteration. This redundancy is tolerated becasue it allows a clean mapping give then
            for i in self.frequency_list: # Need a channel mapping function.
                
                #print(self.frequency_list)
                
                if type(self.M) == type(None):
                    
                    channel_df = self.Sv.Sv[i[0]].to_dataframe(name=None, dim_order=None).reset_index().dropna()   # Keep in mind that the indices are retained with this use of dropna(). I didnt know this for a while.  Its important because the indices associated with values to be incorperated in the clustering model are known and can be copied elsewhere.
                
                else:
                    
                    channel_df = self.Sv.Sv[i[0]].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().dropna()   # Keep in mind that the indices are retained with this use of dropna(). I didnt know this for a while.  Its important because the indices associated with values to be incorperated in the clustering model are known and can be copied elsewhere.
                
                
                #channel_df = self.Sv.Sv[i[0]].to_dataframe(name=None, dim_order=None).dropna() # Convert Sv.Sv[channel] into a pandas dataframe.

                channel_df.rename(columns = {'Sv':str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"}, inplace = True) # Rename the column to the frequency associated with said channel. This value is pulled from the xarray.
                sv_direct_list.append(channel_df[str(self.Sv.Sv[i[0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz"]) # Append columns with each frequency or channel. (It is always best to retain a map between channels and frequencies.)

            self.Sv_df = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None)    
            pre_clustering_df = pd.concat(sv_direct_list, axis = 1).reset_index().select_dtypes(include=['float64']) # Creats a new dataframe from the values previously constructed. This is done to keep it steril.
            
            if type(self.M) == type(None):
                    
                pre_clustering_df["o_index"] = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None).reset_index().dropna().index
                self.Sv_df["o_index"] = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None).reset_index().index  
                                
            else:
                    
                pre_clustering_df["o_index"] = self.Sv.Sv[0].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().dropna().index
                self.Sv_df["o_index"] = self.Sv.Sv[0].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().index 
        
            print(pre_clustering_df)
        
        
        
        if self.model == "ABSOLUTE_DIFFERENCES": # The ABSOLUTE_DIFFERENCES clustering model clusters the absolute value of the differences between a pair permutation of frequency based Sv.Sv values. 
            # Each pair permutation is given it's own column within the dataframe that is fed into KMeans in the same way that with DIRECT each frequency is given it's own column within the dateframe that is fed into KMeans.
            # In other words this model was built to solve the problems of DIRECT by not allowing identical frequencies to be clustered together meaningfully becasue there should no no new information produced by that. 
            # If you attempt to feed the same frequencies in you will get a blank screen. This mean that 100% of the visual information is meaningful.
            sv_absolute_differences_list = []
            for i in self.frequency_pair_combination_list:
                                
                if type(self.M) == type(None):
                    
                    df_1 = self.Sv.Sv[i[0][0]].to_dataframe(name=None, dim_order=None).reset_index().dropna()   # Keep in mind that the indices are retained with this use of dropna(). I didnt know this for a while.  Its important because the indices associated with values to be incorperated in the clustering model are known and can be copied elsewhere.
                    df_2 = self.Sv.Sv[i[1][0]].to_dataframe(name=None, dim_order=None).reset_index().dropna()
                
                else:
                    
                    df_1 = self.Sv.Sv[i[0][0]].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().dropna()   # Keep in mind that the indices are retained with this use of dropna(). I didnt know this for a while.  Its important because the indices associated with values to be incorperated in the clustering model are known and can be copied elsewhere.
                    df_2 = self.Sv.Sv[i[1][0]].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().dropna()
                          
                sv_frequency_absolute_difference_df = (df_1["Sv"] - df_2["Sv"]).abs().values # The principle ABSOLUTE DIFFERENCES equation.

                index_name = "abs(Sv("+str(self.Sv.Sv[i[0][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz" +")-Sv("+ str(self.Sv.Sv[i[1][0]].coords.get("channel")).split("kHz")[0].split("GPT")[1].strip()+"kHz))"
                
                x = pd.DataFrame(data = sv_frequency_absolute_difference_df, columns = [index_name])
                
                sv_absolute_differences_list.append(x[index_name])
                


            self.Sv_df = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None)
            pre_clustering_df = pd.concat(sv_absolute_differences_list, axis = 1).reset_index().select_dtypes(include=['float64']) # Version 1 

            
            if type(self.M) == type(None):
                
                pre_clustering_df["o_index"] = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None).reset_index().dropna().index
                self.Sv_df["o_index"] = self.Sv.Sv[0].to_dataframe(name=None, dim_order=None).reset_index().index  
            
            else:
                
                pre_clustering_df["o_index"] = self.Sv.Sv[0].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().dropna().index
                self.Sv_df["o_index"] = self.Sv.Sv[0].where(~self.M.isnull()).to_dataframe(name=None, dim_order=None).reset_index().index  
            
        return pre_clustering_df # Returns pre-clustering dataframe.


    def assign_sv_clusters(self):
        """_summary_
        """        
        
        
        # Before cropping prepares records to be fed into KMeans.
        
        self.pre_clustering_df = self.construct_pre_clustering_df() # Construct dataframe which is to be fed into the kmeans clustering algorithm. In this each column is a frequency.
        

        # Normalizes pre-clustering-dataframe.
        
        print(self.pre_clustering_df.drop(["o_index"], axis = 1))
        self.normalized_pre_clustering_df = pd.DataFrame(preprocessing.scale(self.pre_clustering_df.drop(["o_index"], axis = 1))) # Make into a dataframe.

        
        # Run clustering on normalized-pre-clustering-dataframe.
        
        kmeans = KMeans(n_clusters=self.cluster_count, random_state = self.random_state, init='k-means++', n_init=10, max_iter=300)  # Kmeans configuration object.
        X = self.normalized_pre_clustering_df.values # 'X' is the sklearn convention. 
        clustered_records = kmeans.fit_predict(X) # The clustering data in df format.
        
        
        clustered_records_df = pd.DataFrame()
        clustered_records_df[self.frequency_set_string] = clustered_records + 1
        clustered_records_df["o_index"] = self.pre_clustering_df["o_index"]
        clustered_records_df["ping_time"] = self.Sv_df.reset_index()["ping_time"]

        
        self.clustered_records_df = pd.merge(self.Sv_df, clustered_records_df, how='left', on=['o_index']) # TODO : The 'new' procedure which permits .EVR file interpretation.
        self.clustered_records_df = self.clustered_records_df.drop(["Sv"], axis=  1)
        self.clustered_records_df = self.clustered_records_df.drop(["o_index"], axis=  1)
        self.clustered_records_df.index = self.Sv_df.index
        self.clustered_records_df = self.clustered_records_df.drop(["ping_time"], axis=  1)
        
        
        self.clustered_records_da = self.clustered_records_df.to_xarray()
        
        
        km_cluster_maps = [] # An array which helps prepare the dataArray dimensionality. It will store copies of clustering data. The redundancy may seem unfortune but it is infact nessecary due to the nature of how the cluster map was created. 
        

        plt.figure(figsize=(22,8))
        
        plt.title("Cluster Map Ratio Profile") # Plot configuration.
        (self.clustered_records_df[self.frequency_set_string].value_counts(normalize=True) * 100).plot.pie(cmap='jet', labels=self.clustered_records_df[self.frequency_set_string].value_counts(normalize=True).index, autopct='%1.1f%%')

        
        for i in range(len(self.Sv.Sv)): # For each frequency, map an equivilent clustering set to satisfy dimensionality constraints. This is nuanced but nessecary to make the DataArray meaningful and sensable.
            #TODO: I know km_cluster_maps seems like a strange object and looks strange but it is not trust me. It is deeply needed to organize the information into the xarray.
            # The reason this was done was this way was due to the way the model was devised and dimensionality requirements, 
            # but in the end we couldnt treat a cluster map as a simple frequency per se which can be organized into Sv.Sv[channel] 
            # even though it may be elegent this way however satisfying dimensionality requriements with respect to the xarray usecase seemed more essential.
            # Instead just like the water level or something this cluster map is placed a higher level up but copies exist which map loosly speaking to the channels aformentioned.
            km_cluster_maps.append(self.clustered_records_da[self.frequency_set_string].values) # Multiply this entry for each channel becasue it needs to be mapped meaningfully to match the dimensionality of range_sample and ping_time.
       
        
        self.Sv["km_cluster_map"+self.frequency_set_string] = xr.DataArray(
            data = km_cluster_maps, dims = ['channel','ping_time','depth'], # Clustering data is appended with respect to ping_time and range_sample.
            attrs = dict( # Atrrributes as a dictionary.
                description = "The kmeans cluster group number.", # Assigns a description attribute to the DataArrray.
                units = "Unitless", # Assigns a physical unit attribute to the DataArray.
                clusters = self.cluster_count, # Assigns a cluster count variable 'k' attribute to the DataArray.
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
    def __init__(self, EK_data_path, EK_data_filenames, save_path, sonar_model, frequency_list, cluster_count, random_state = None, n_init = 10, max_iter = 300, model = "DIRECT", cluster_color_map = "viridis", echogram_color_map = "viridis", plot = True, data_reduction_type = None, range_meter_bin = None, ping_time_bin = None, range_sample_num = None, ping_num = None, remove_noise = True, ping_time_begin = None, ping_time_end = None, range_sample_begin = None, range_sample_end = None, line_files = None, region_files = None ):
           
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
        self.EK_data_path = EK_data_path
        self.EK_data_filenames = EK_data_filenames
        
        #self.file_path = file_path # The .raw or .nc filepath.
        
        
        
        
        #self.file_name = self.file_path.split("."+self.file_path.split(".")[-1])[0].replace(".","").split("/")[-1] # The filename without the path prefix or extension.
        self.save_path = save_path # The save path of the echograms, cluster maps and other resources.
        
        # KMeans configuration
        
        self.frequency_list = frequency_list
        self.cluster_count = cluster_count # The value 'k'. the integer number of clusters ( or colors ) the plot will display.
        self.random_state = random_state
        self.model = model # Preclustering model.
        self.n_init = n_init
        self.max_iter = max_iter # 
        
        self.cluster_color_map = cluster_color_map # Color profile of echogram and cluster map plots.
        self.echogram_color_map = echogram_color_map # Color profile of echogram and cluster map plots.

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

        self.frequency_list_string = self.__construct_frequency_list_string(self.frequency_list)
        logger.info("Frequency List String : "+self.frequency_list_string)
         
        self.sonar_model = sonar_model # The Echosounder model string which Echopype requires as base parametrization when converting .raws or .nc files.
        
        self.line_files = line_files # Line files list. Files are provided from echoview.
        self.region_files = region_files # Region files list. Files are provided from echoview.
        self.M = None
        
        
    

              
        for EK_file in EK_data_filenames:
            
            
            
            self.__construct_kmeans_cluster_map(self.EK_data_path+"/"+EK_file) # Main execution function encapsulating nessecary variable constructions.
        
        
        
    
                
    def __construct_kmeans_cluster_map(self, file_path):
        """Once this class object is instantiated, a run function is employed to perform some required initialization sequences so that all class object variables can be constructed. 
        Another way to think about it is that run() handles the top level abstraction of KMeanClusterMap() managment and Sv prep for KMeans clustering analysis. 
        This involves converting, cropping, dropping Nans, computing MVBSusing physical units or sample number, 
        removing noise and applying .EVL or .EVR files.
        """        
        file_name = self.__construct_file_name(file_path)
        
        if self.__vet_file_path(file_path) == True:


            self.__construct_save_directory(self.save_path)
            
            #self.ed = self.__get_combined_ed("nc_files")
            
            self.Sv = self.__compute_Sv(file_path, self.sonar_model)    
            self.frequency_map = self.__construct_frequency_map(self.Sv, self.frequency_list)
        
            logger.info("[ CHANNEL, FREQUENCY ] Map : "+str(self.frequency_map))        
            logger.info("Clustering Frequency Set String : "+self.__construct_frequency_set_string(self.frequency_map))

            
            
            self.Sv = self.__rectangular_region_of_interest(self.Sv, self.ping_time_begin, self.ping_time_end, self.range_sample_begin, self.range_sample_end )
            self.Sv = self.__drop_nans(self.Sv)
            if self.remove_noise == True and self.range_sample_num != None and self.ping_num != None: 
                self.Sv = self.__remove_noise(self.Sv, self.range_sample_num, self.ping_num)     
            if self.range_meter_bin != None and self.ping_time_bin != None:               
                self.Sv = self.__configure_PU_MVBS(self.Sv, self.range_meter_bin, self.ping_time_bin)                   # Process routine. This deals with MVBS parametrization.       
            if self.ping_num != None and self.range_sample_num != None:          
                self.Sv = self.__configure_SN_MVBS( self.Sv, self.ping_num, self.range_sample_num )                             
            
            
  
            
            
            self.Sv = self.__assign_depth_coordinates(self.Sv)
            
            

            
            
            if type(self.M) == type(None):
            
                for i in range(len(self.region_files)):

                    if i == 0:
                        
                        self.M = self.__process_region_files(self.Sv, self.region_files[i])
                        
                    else:
                
                        self.M = self.__process_region_files(self.Sv, self.region_files[i], self.M)
                        
            else:
                
                for i in range(len(self.region_files)):
                        
                    self.M = self.__process_region_files(self.Sv, self.region_files[i], self.M)




            if type(self.M) == type(None):    
                
                
                for i in range(len(self.line_files)):
                        print("Line File : "+ str(i) )
                        if i == 0:
                            
                            self.M = self.__process_line_files(self.Sv, self.line_files[i]) 

                        else:
                    
                            self.M = self.__process_line_files(self.Sv, self.line_files[i], self.M)
            
            else:
                
                for i in range(len(self.line_files)):
                        
                    self.M = self.__process_line_files(self.Sv, self.line_files[i], self.M)
                    
                    
                    
            
            self.Sv = self.__assign_kmeans_cluster_map(self.Sv, self.frequency_list, self.cluster_count, self.random_state, self.n_init, self.max_iter, self.model, self.M)    

            if self.save_path != None: # If a save path was provided.                 
                self.__full_save(self.Sv, file_path, self.frequency_list, self.frequency_map, file_name) # This saves the kmeans cluster map and a corresponding echogram for each involved frequency.
            if self.plot == True:      
                self.__plot_figures()

                
                
                
                
                
    def __get_combined_ed(self, directory):
        
        from dask.distributed import Client
        client = Client()
        ed_list = []
        for converted_file in os.listdir(directory):
            if converted_file in self.EK_data_filenames:
                ed_list.append(ep.open_converted(converted_file))  # already converted files are lazy-loaded
        combined_ed = ep.combine_echodata(
        ed_list, 
        zarr_path='combined_echodata/combined_echodata.zarr', 
        client=client )
        
        return combined_ed
        
                    
                
    def __construct_file_name(self, file_path):
        return file_path.split("."+file_path.split(".")[-1])[0].replace(".","").split("/")[-1] # The filename without the path prefix or extension.
        
    def __construct_frequency_set_string(self , frequency_list):
        """So the idea behind a frequency_set_string is to serve as a quick and simple representative string which describes the kmeans clustering map and the frequency data that was employed to create it.
        """        
        frequency_set_string = "" # Declare frequency_set_string. We call it a set even though it is a list is becasue it is not meant to change but the list aspect was useful because it allowed mutability.
        for i in frequency_list: # For each frequency in the frequency_list.
            frequency_set_string += i[1].split(" ")[0]+"kHz,"

        frequency_set_string = "<("+frequency_set_string+"," # Start defining the frequency_set_string.
        frequency_set_string = frequency_set_string.split(",,")[0]+")>" # Finishing defining the frequency_set_string.
        return frequency_set_string
        
    def __construct_frequency_map(self, Sv, frequency_list):

        frequency_map = [] # Declare a frequency list to be populated with string frequencies of the form [[1,'38kHz'],[2,'120kHz'],[4,'200kHz']] where the first element is meant to be the channel representing the frequency. This is an internal object. Do not interfere.
        for j in frequency_list: # For each frequency 'j'.
            for i in range(len(Sv.Sv)): # Check each channel 'i'.
                if str(Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip() == j.split("kHz")[0].strip(): # To see if the channel associates with the frequency 'j' .
                    frequency_map.append([i,str(Sv.Sv[i].coords.get("channel")).split(" kHz")[0].split("GPT")[1].strip()+" kHz"]) # If so append it and the channel to the 'frequency_list'.
        return frequency_map
                
                
    def __construct_save_directory(self, save_path):
        
        if not os.path.exists(save_path): # If directory does not exist.
            
            os.makedirs(save_path)
                
    def __vet_file_path(self, file_path):
        
        if file_path is None or ( file_path.split(".")[-1] != "nc" and file_path.split(".")[-1] != "raw" ):   
            
            logger.error("Provide valid .raw or .nc file")     
            return False
        
        else:
            
            return True
            
    def __assign_kmeans_cluster_map(self, Sv, frequency_list, cluster_count, random_state, n_init, max_iter, model, M = None): 
               
        logger.info('Preparing KMeans clustering.') # Logging message.
        kmeans_operation = KMeansOperator( Sv = Sv,  M = M, frequency_list = frequency_list, cluster_count = cluster_count, random_state = random_state, n_init = n_init, max_iter = max_iter, model = model )  




        return kmeans_operation.Sv
            
            
    def __plot_figures(self):
        
        logger.info('Plotting KMeans cluster map and corresponding echograms...')    # Logging message.    
        plt.show()
                
                
    def __compute_Sv(self, file_path, sonar_model):
        
        if file_path.split(".")[-1] == "raw": # If sonar data file provided was a .raw file.

            logger.info('Attempting to convert (' + file_path + ") to a .nc format...")                                    # Logging message.
            
            ed = ep.open_raw(raw_file = file_path, sonar_model = sonar_model) # Principle echopype conversion function. This provides echodata object.
            logger.info('Saving .nc file converted from .raw format as (' + "/nc_files" + file_path + ")...")     # Logging message.
            
            
            ed.to_netcdf(save_path='./nc_files')
            
            
            Sv = ep.calibrate.compute_Sv(ed)
            logger.info("Converted!")
            
            return Sv

        if self.file_path.split(".")[-1] == "nc": # If a .nc file is provided by user.

            ed = ep.open_converted(self.file_path) # Principle echopype conversion function. This provides echodata object.
            self.Sv = ep.calibrate.compute_Sv(ed)
        
            return Sv
        
    def __process_line_files(self, Sv, line_file_json, M = None):
        
            logger.info("Reading line files.")
            
            
            # read an example .evl file
            line = er.read_evl(line_file_json["path"])
            new_M = line.mask(Sv.Sv.isel(channel=0).drop('channel'))
            
            
            
            
            if line_file_json["depth_offset"][0] > 0 and line_file_json["depth_offset"][1] == "above":
            
            
                new_M = new_M.shift(depth = line_file_json["depth_offset"][0])
                new_M = new_M.fillna(1)
            
            if line_file_json["depth_offset"][0] < 0 and line_file_json["depth_offset"][1] == "above":
            
                new_M = new_M.shift(depth = line_file_json["depth_offset"][0])
                print(new_M)
                new_M = new_M.where(new_M != 0, np.nan)
                print(new_M)
            
            if line_file_json["depth_offset"][0] > 0 and line_file_json["depth_offset"][1] == "below":
            


                new_M = new_M.shift(depth = line_file_json["depth_offset"][0])
                new_M = new_M.fillna(1)
                new_M = new_M.where(new_M != 1, 2)
                new_M = new_M.where(new_M != 0, 1)
                new_M = new_M.where(new_M != 2, np.nan)

                
            
            if line_file_json["depth_offset"][0] < 0 and line_file_json["depth_offset"][1] == "below":
            
                new_M = new_M.shift(depth = line_file_json["depth_offset"][0])
                new_M = new_M.fillna(0)

            
            
            if type(M) == type(None): # If this is the first 'Mask' to be created.
                
                M = new_M # Set the final 'Mask' to be this new one if there was no other preceding it.

                return M # Return M
            
            else:
            
                
                M = M.fillna(0) # Fill NaNs with Zero so that algebra can be done on Mask records. (These will eventually be turned into NaNs again but some Mask algebra needs to be done in order to collect similar regions and place them in similar catigories.)
                new_M = new_M.fillna(0) # Fill NaNs in new mask with zero.    
                M = M + new_M # Sum the old and new masks.
                M = M.where(M == 0, np.nan) # Where zero remains, no regions of interest are present or represented.
                print(M)
                return M
            
            
            
        
    def __assign_depth_coordinates(self, Sv):
        

            # create depth coordinate:
            echo_range = Sv.echo_range.isel(channel=0, ping_time=0)
            # assuming water levels are same for different frequencies and location_time

            depth = Sv.water_level.isel(channel=0, time3=0) + echo_range
            depth = depth.drop_vars('channel')
            # creating a new depth dimension

            Sv['depth'] = depth
            Sv = Sv.swap_dims({'range_sample': 'depth'})
            
            return Sv
            
    def __process_region_files(self, Sv, file_path, M = None):
        

            logger.info("Reading region files.")
            
            
            EVR_FILE = file_path
            
            r2d = er.read_evr(EVR_FILE)

            #print(r2d.data["region_name"])
            if type(M) == type(None):
                
                for i in range(len(r2d.data.region_id.values)):
                    
                    if i == 0:
                        
                        M = r2d.mask(Sv.isel(channel=0).drop('channel'), [i+1], mask_var="ROI") # Convert nans to zero so a summing interpretation of the mask can be utilized. This allows multiple regions to be combined.
                    
                    else:
                        
                        M = M.combine_first(r2d.mask(Sv.isel(channel=0).drop('channel'), [i+1], mask_var="ROI"))
                print(M)
                return M
            
            else:
            
            
                for i in range(len(r2d.data.region_id.values)):
                    
                    M = M.combine_first(r2d.mask(Sv.isel(channel=0).drop('channel'), [i+1], mask_var="ROI"))
                    
                print(M)
                return M
            


           
            
    def __rectangular_region_of_interest(self, Sv, ping_time_begin, ping_time_end, range_sample_begin, range_sample_end ):
        
        logger.info("Rectangular region of interest is cropped out of analysis region.")
        Sv_RROI = Sv.isel(range_sample=slice(range_sample_begin, range_sample_end), ping_time=slice(ping_time_begin, ping_time_end))
        logger.info("Region Cropped!")
        return Sv_RROI        
    
    def __drop_nans(self, Sv):
        
        logger.info("Dropping NaNs") 
        Sv_naNs_dropped = Sv.dropna(dim="range_sample", how="any") # Pulls a cleaner version of Sv without NaNs.
        logger.info("NaNs Dropped!")
        return Sv_naNs_dropped
    
    def __remove_noise(self, Sv, range_sample_num, ping_num ):
                    
        logger.info('Removing noise from Sv...')                            # Logging message.
        logger.info('   range_sample = ' + str(range_sample_num))      # Logging message.
        logger.info('   ping_num = ' + str(ping_num))                  # Logging message.
        logger.info("Noise Removed!")

        return ep.clean.remove_noise(Sv, range_sample_num = range_sample_num, ping_num = ping_num)  
      
    def __configure_PU_MVBS(self, Sv, range_meter_bin, ping_time_bin):
        """Configure MVBS using provided class variables. This internal method should not be directly employed by user.
        """

        logger.info('Calculating MVBS using reduction by physical units.')  # Logging message.
        logger.info('   range_meter_bin = ' + str(range_meter_bin))    # Logging message.
        logger.info('   ping_time_bin = ' + str(ping_time_bin))        # Logging message.
        self.Sv = ep.commongrid.compute_MVBS(Sv, range_meter_bin = range_meter_bin, ping_time_bin = ping_time_bin )
                
        return Sv
    
    def __configure_SN_MVBS(self, Sv, range_sample_num, ping_num):
        """Configure MVBS using provided class variables. This internal method should not be directly employed by user.
        """
                
        if ping_num != None and range_sample_num != None:
            logger.info('Calculating MVBS using reduction by sample number.')
            logger.info('   range_sample_num = ' + str(range_sample_num))
            logger.info('   ping_num = ' + str(ping_num))
            self.Sv = ep.commongrid.compute_MVBS_index_binning( Sv, range_sample_num=range_sample_num, ping_num=ping_num )
            
        return Sv
            
    def __save_echogram(self, data_array, file_path, channel):
        """_summary_

        Args:
            data_array (xarray.DataArray): For a given channel, save echogram to save_path for the corresponding frequency if it exists.
            channel (str): Integer represented as a string. Example: "3" 
        """        
        plt.figure(figsize=(22,8)) # Instantiates a matplotlib figure object which is required for plotting multiple figures. Required for plotting and saving. Recalled when plt.show() is called.
        echogram_cmap = plt.get_cmap(self.echogram_color_map) # The plots color scheme. There are various options. See above.     
        
        
        data_array[channel].transpose("depth","ping_time").plot(cmap = echogram_cmap)
        
        
        plt.title("frequency = "+self.__get_frequency(channel)+",\nfile = "+file_path+",\ncolormap = "+self.echogram_color_map ) # Plot configuration.    
        plt.gca().invert_yaxis() # Since the image flips upside down for some reason, this is applied as a correction.
        plt.savefig(fname = self.save_path+"/eg:"+self.__construct_file_name(file_path)+"<"+self.__get_frequency(channel)+">", dpi=2048) # Save a parametrized filename. # File naming/saving configuration.  
    
        
    def __save_cluster_map(self, Sv, file_path, frequency_list, frequency_map, file_name):
        """Saves the Kmeans cluster map.

        Args:
            Sv (_type_): Principle xarray object 
        """        
        
        plt.figure(figsize=(22,8)) # Instantiates a matplotlib figure object which is required for plotting multiple figures. Required for plotting and saving. Recalled when plt.show() is called.
        cluster_cmap = plt.get_cmap(self.cluster_color_map, self.cluster_count) # The plots color scheme. There are various options. See above.




         
        Sv["km_cluster_map"+self.__construct_frequency_set_string(frequency_map)][0].transpose("depth","ping_time").plot(cmap = cluster_cmap)
        plt.title(self.__construct_frequency_set_string(frequency_map)+",\ncluster_count = "+str(self.cluster_count)+",\nrandom_state = "+str(self.random_state)+",\nfile = "+file_path+",\ncolormap = "+self.cluster_color_map) # Plot configuration.
        plt.gca().invert_yaxis() # Since the image flips upside down for some reason, this is applied as a correction.
        plt.savefig(self.save_path+"/km:"+file_name+"<"+ self.__construct_frequency_list_string(frequency_list)+"k="+str(self.cluster_count)+"_rs="+str(self.random_state)+"_cm="+self.cluster_color_map+"_md="+str(self.model)+"_rmb="+str(self.range_meter_bin)+">", dpi=2048) # File naming/saving configuration.


    def __full_save(self, Sv, file_path, frequency_list, frequency_map, file_name):
        """Saves a KMeans cluster map and associated echograms to the designated directory defined by save_path parameter.

        Args:
            Sv ((Modified Echodata Object) xarray.DataArray): Echodata object. Principle object of echopype library.
        """ 
        logger.info('Saving cluster map and corresponding echograms...')    # Logging message.       
        self.__save_cluster_map(Sv, file_path, frequency_list, frequency_map, file_name) # Save the kmeans cluster Map image.
        for frequency in frequency_map: # For each frequency record within the frequency list. Example : [1, '38 kHz'], [2, '120 kHz'], [0, '18 kHz'], [3, '200 kHz']
            self.__save_echogram(Sv["Sv"], file_path, frequency[0]) # Save the echogram image.

    def __construct_frequency_list_string(self, frequency_list):
        """Returns a string which serves as a representation of which frequencies are utilizednin KMeans. Example : "38kHz_70kHz_120kHz_200kHz_". For use in file and variable naming mostly.

        Returns:
            str: Example : "38kHz_70kHz_120kHz_200kHz_"
        """        
        frequency_list_string = ""
        for frequency in frequency_list:
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
                
                cluster_color_map = data["plotting"]["cluster_color_map"],                          # Color map variable. Defaults to "viridis" Check out https://matplotlib.org/stable/tutorials/colors/colormaps.html to see options. Examples include 'jet', 'plasma', 'inferno', 'magma', 'cividis'.
                echogram_color_map = data["plotting"]["echogram_color_map"],
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