from UT_GLOBUS import downloader
from UT_GLOBUS import processing
from UT_GLOBUS import globus2wrf
from UT_GLOBUS import ut_globus
import os
import ee
import wxee
import subprocess
print('Imports successful')

############### Inputs ##########################
esd_username = ''                #### NASA Earthdata username
esd_email = ''                    #### NASA email ID                   
city = 'Austin'
x_min, x_max, y_min, y_max = -98.000,-97.6079,30.3392,30.4334
path = '/home/globus/GLOBUS/Example'
OSM_path = '/home/globus/GLOBUS/Dependency/OSM'
lastools = '/home/globus/GLOBUS/Dependency/LAStools/bin'
rf_model = '/home/globus/GLOBUS/Dependency/RF/UT-GLOBUS.joblib'
g2w_path = '/home/globus/GLOBUS/Dependency/globus2WRF'

############### Google Earth Engine ##########################
subprocess.run('earthengine authenticate', shell=True, stdout=subprocess.PIPE)
ee.Initialize()
wxee.Initialize()

############### Download ##########################
##ICESat-2 - Tested
downloader.icesat_2(x_min,y_min,x_max,y_max,path,city,esd_username,esd_email)
## Microsoft footprints - Tested
downloader.Microsoft_Footprints(x_min,y_min,x_max,y_max,path,city)
#OpenStreetMaps -Tested
downloader.osm(x_min,y_min,x_max,y_max,path,OSM_path,city)
##ALOS - SRTM (nDSM) - Tested
downloader.alos(x_min,y_min,x_max,y_max,path,city)
##GEDI - Tested
downloader.gedi(x_min,y_min,x_max,y_max,path,city)
## ESA worldcover - Tested
downloader.esa(x_min,y_min,x_max,y_max,path,city)
## Population - Tested
downloader.population(x_min,y_min,x_max,y_max,path,city)

############### Process ##########################

## OSM - Tested
processing.osm_processing(path+'/'+city,city)
## ICESat-2 (Phoreal) - Tested
processing.phoreal(path,city)
##Grid ICESat-2 photons - Tested
processing.txt2grid(lastools,path+'/canopy',path+'/ground')
## ICESat-2 nDSM - Tested
processing.icesat_ndsm(path+'/canopy/canopy.tif',path+'/ground/ground.tif',path)

## Merged ICEsat-2 and GEDI - Tested
myfile = open(path+'/merge.txt', 'w')
string1 = str(path+'/icesat-2_ndsm.tif')
myfile.write("%s\n" % string1)
string2 = str(path+"/GEDI.nc")
myfile.write("%s\n" % string2)
myfile.close()
processing.icesat_gedi(path+'/merge.txt',path+'/merged.tif')
## Altimetry raster to vector - Tested
processing.to_pixels(path+'/merged.tif',path)
## Interpolation - Tested
processing.triangular(path,'vector.gpkg',path)
## Spaceborne nDSM - Tested
processing.space_ndsm(path+"/"+"ALOS.nc",path+'/GEDI.tif',path)
##Population - Tested
processing.population(path+'/Pop.nc',path+'/GEDI.tif',path)

############### Pre-process geometries - Tested ##########################
processing.fix_geometry(path+'/'+city+'_MS'.replace(" ", "_")+'.gpkg',path,city+'_MS_fixed.gpkg')
print('Fixed MS')
processing.fix_geometry(path+'/'+city+'/'+city+'.gpkg',path,city+'_OSM_fixed.gpkg')
print('Fixed OSM')
##processing.merge_geomtery(path+'/'+city+'_MS_fixed.gpkg',path+'/'+city+'_OSM_fixed.gpkg',path)
#print('Merged')
##processing.fix_geometry(path+'/ms_osm.gpkg',path,city+'_MS_OSM_fixed.gpkg')
#print('Fixed Geometries')
##processing.dissolve(path+'/'+city+'_MS_OSM_fixed.gpkg',path)
#print('Dissolved')
##processing.single2multi(path+'/dissolved.gpkg',path)
#print('Converted to singlepart')
##processing.delete_holes(path+'/multi.gpkg',path)
#print('Filled holes')

############### Tabular data - Tested ##########################

processing.zonal_pass1(path+'/'+city+'_MS_fixed.gpkg',path+'/space.tif',path)
#print('Zonal pass 1')
processing.zonal_pass2(path+'/Zonal_pass1.gpkg',path+'/pop.tif',path)
print('Zonal pass 2')
processing.zonal_pass3(path+'/Zonal_pass2.gpkg',path+'/'+city+'_OSM_fixed.gpkg',path)
print('Zonal pass 3')

################ RF model #########################
ut_globus.ut_globus(path+'/osm_joined.gpkg', path, city, rf_model)

############## GLOBUS to WRF #####################
globus2wrf.g2w(g2w_path, path, path+'/Austin.gpkg', path+'/ESA.nc', city)
