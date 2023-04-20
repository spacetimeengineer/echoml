# Acoustic Analysis ML Project

A package dedicated to an active research and development operation tasked with applying machine learning algorithms to the data produced by underwater sonar/tranduceer systems (EK60, EK80, ect). The main purpose of these sonars is to essentially take pictures of the sea bed and the local fauna and flora of that enviornment. One of the difficulties of such a task is this dilema of identification. Since we humans can only see with light and not sound and despite the accurate and high resolution data produced by sonars the problem is in the interpretation of said data which is something that may greatly utilize machine learning. One of the constraints imposed by the transducers is that there are only a handful of frequencies which are utilized. They all ping at the same time but each can really only register object which posess sizes comparible to the wavelength of the sound waves which pick it up. The mystery which these algorithms are being optimized to uncover is the information which may not be picked up by any of the sonar transducers due to size or density constraints but instead leave traces or signatures across multiple frequencies that are discernable from noise which only can be picked up said algoithms. 


# This project has Identified 2 classes of acoustic analysis tasks.

    # A 3D Plot on with X , Y , Z Axis
    # A 2D Plot on Z , T Axis

Within each of these classes there will bwe numerous possible implementations on KMEANS. Principly through the use of plotting kmeans with the differences in dB reading between channels.


### How is the K-Means Algorithm employed. What does it operate on?
    
    A K-Means agent will be used to the individual differences in 






# Building Enviornment

1.) It is reccomended that you prepare a virtual environment when working with echopype becasue it does not entirely appear stable or plug-in-play yet.

    $ pipenv install matplotlib numpy echopype

# Logging Into Repository



# Commiting Code (Simple Version)

      $ git pull  
      $ git add .
      $ git commit -m "commit message goes here"
      $ git push


# How to prepare ML algorithm

Imagine we are strating with a query matrix of the kind used in the previous iteration of this package which uses csvs.


        # Get some complex query and produce a local database or something.
        local_db = "fmrd_test_database.db"
        #reef_db = "reef.db"

        # Create cleaned database.
        query_matrix = {

            "database_path" : local_db,
            "from" : "fmrd_test_dataset",
            "where" : "duration < 1000",
            "kmeans_select": 
            {

                "TARGSPEC1",
                "KEPT",
                "DISCARD"
            },
            "axis_select" : 
            {
                "LAT",
                "LON",
                "TOTAL"

            },
            "k" : 10
        }

        # Create agent objects.
        kmeans_01 = KMeansAgent(query_matrix)
        # Plot these functions.
        kmeans_01.plot()


Now the trick for application on Sv is to interpret the columns as channels. Next, consider one main difference between Sv and a database table is that records are essentially multi dimensional. The custering still need to be resolved on a single dimension individually but the representation must be such that it mapps back to the original record set. For this reason a new xarray is created and there is a class structure dedicated to the creation of new Sv objects which are identical to the original but have added a new dataarrays with clustering information. THis information is mapped to the original record by being part of this new xarray.


        modified_Sv = Sv_Ml(k = 5, kmeans_channels = [0,3,4])
        # Now plot modified_Sv.Sv_ml into whatever function processes the original data.
        
