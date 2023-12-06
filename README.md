# The University of Texas - GLObal Building heights for Urban Studies (UT-GLOBUS)
The primary intention of UT-GLOBUS is to provide data for urban modeling. UT-GLOBUS does not aim to provide accurate building-level information but to enable calculation of urban canopy parameters for urban energy balance and weather models. This package uses Microsoft openstreetmaps building footprints. 

Installation notes: <br>
git lfs clone https://github.com/texuslabut/UT-GLOBUS/ <br>
Install the ut-globus anaconda environment using the '.yml' file provided in UT-GLOBUS/GLOBUS. <br>
Install PhoREAL (https://github.com/icesat-2UT/PhoREAL). <br>
Download LASTools (https://github.com/LAStools/LAStools) in UT-GLOBUS/GLOBUS/Dependency/ folder and make. <br>
To output the urban canopy parameters in WRF-Urban binary files, gfortran and gcc are required to be installed. <br>

Usage notes: <br>
NASA Earthdata account is required to download ICESat-2 data and Google Earth Engine needs to authenticated. <br>
Fill in the 'inputs' section in example.py. Make sure to set the paths for dependencies correctly. <br>
Run the 'example.py'. <br>


