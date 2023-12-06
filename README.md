# The University of Texas - GLObal Building heights for Urban Studies (UT-GLOBUS)
The primary intention of UT-GLOBUS is to provide data for urban modeling. UT-GLOBUS does not aim to provide accurate building-level information but to enable calculation of urban canopy parameters for urban energy balance and weather models. This package uses Microsoft openstreetmaps building footprints. 

Installation notes: <br>
1. git lfs clone https://github.com/texuslabut/UT-GLOBUS/ <br>
2. Install the ut-globus anaconda environment using the '.yml' file provided in UT-GLOBUS/GLOBUS. <br>
3. Install PhoREAL (https://github.com/icesat-2UT/PhoREAL). <br>
4. Download LASTools (https://github.com/LAStools/LAStools) in UT-GLOBUS/GLOBUS/Dependency/ folder and make. <br>
5. To output the urban canopy parameters in WRF-Urban binary files, gfortran and gcc are required to be installed. <br>

Usage notes: <br>
1. NASA Earthdata account is required to download ICESat-2 data and Google Earth Engine needs to authenticated. <br>
2. Fill in the 'inputs' section in example.py. Make sure to set the paths for dependencies correct. <br>
3. Run the 'example.py'. <br>
4. All the data will be saved in the path provided. In this case, in Example folder. Building-level data is in the file 'city.gpkg' ('Austin.gpkg' in the example). <br>
5. Follow the text file provided to ingest data into WRF-Urban to run the multi-layer BEP-BEM model.


