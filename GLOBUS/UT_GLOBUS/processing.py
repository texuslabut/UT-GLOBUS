import phoreal as pr
import re
from qgis.core import *
from qgis.gui import *
from qgis.PyQt.QtWidgets import *
import processing
from qgis.analysis import QgsNativeAlgorithms
from processing.core.Processing import Processing
Processing.initialize()
import pandas as pd
import geopandas as gpd
import shapely.geometry
import mercantile
from tqdm import tqdm
import os
import tempfile
import fiona
from shapely.geometry import shape
import glob
import numpy as np
import rioxarray as rio
import icepyx as ipx
import xarray as xr
import h5py
import json
from pprint import pprint
import time
import urllib as urllib
from osgeo import gdal,ogr
from qgis.core import QgsVectorLayer
from shapely import wkt
import re
import xarray
import geopandas
import ee
import wxee
from qgis.core import QgsProject
from qgis.PyQt.QtCore import QFileInfo


def osm_processing(root,city_name):
    path = root+"/"
    print('Processing OSM files in '+root)
    data = geopandas.read_file(path+'multipolygons.shp')
    #geometry__ = []
    levels = []
    height = []
    height_2 = []
    building__ = []
    for k in range(0,len(data)):
        #geometry__.append(data.geometry[k].to_wkt())

        try:
            levels.append(np.ceil(float(re.sub('[^\d\.]', '', data.bld_levels[k]))*4))
        except:
            levels.append(np.nan)

        try:
            height.append(float(re.sub('[^\d\.]', '', data.bld_hght[k])))
        except:
            height.append(np.nan)

        try:
            height_2.append(float(re.sub('[^\d\.]', '', data.height[k])))
        except:
            height_2.append(np.nan) 

        building__.append(data.building[k])
        
    geometry__ = data.geometry.to_wkt()
    df = pd.DataFrame({'geom': geometry__,'levels': levels, 'height':height,'height2':height_2,'building':building__})
    df['geom'] = df['geom'].apply(wkt.loads)
    df[df['levels']>500]= np.nan
    df = df[df['building'].notna()]
    df = df.drop(columns=['building'])
    temp = df.drop(columns=['geom'])
    df['OSM'] = temp.mean(axis=1)
    gdf = geopandas.GeoDataFrame(df, geometry='geom',crs='EPSG:4326')
    gdf.to_file(path+city_name+'.gpkg', driver='GPKG') 


def sorted_nicely(l): 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def phoreal(path, city_name):
    pairs = ['gt1r','gt2r','gt3r','gt1l','gt2l','gt3l']
    print('Processing ICESat-2 data in '+path)
    path_alt03 = path+'/icesat2_'+city_name+'/ATL03/'
    path_alt08 = path+'/icesat2_'+city_name+'/ATL08/'

    os.mkdir(path+'/canopy')
    os.mkdir(path+'/ground')
    
    files_atl03 = os.listdir(path_alt03)
    files_atl08 = os.listdir(path_alt08)

    files_h5 = [f for f in files_atl03 if f[-2:] == 'h5']
    files_h5 = sorted_nicely(files_h5)

    for files_ in files_h5:
        atl03_file = path_alt03+str(files_)
        atl08_file = [m for m in files_atl08 if m.startswith('processed_ATL08_') and m.endswith('.h5') and str(atl03_file[-33:-10]) in m]
        if atl08_file:
            for beam in pairs:
                try:
                    atl03 = pr.reader.get_atl03_struct(atl03_file, beam, path_alt08+atl08_file[0])
                    data = atl03.df
    
                    df_canopy = data.loc[(data['classification'] == 3) & (data['signal_conf_ph']>=3)]
                    df_ground = data.loc[(data['classification'] == 1) & (data['signal_conf_ph']>=3)]
    
                    canopy = df_canopy[['lon_ph','lat_ph','h_ph']].copy()
                    ground = df_ground[['lon_ph','lat_ph','h_ph']].copy()
    
                    canopy = canopy.reset_index(drop=True)
                    ground = ground.reset_index(drop=True)
    
                    canopy.to_csv(path+'/canopy/'+str(atl08_file[0])+'_'+str(beam)+'.txt' ,index=None, sep='\t', mode='a')
                    ground.to_csv(path+'/ground/'+str(atl08_file[0])+'_'+str(beam)+'.txt' ,index=None, sep='\t', mode='a')
                    
                except:
                    pass

def las2grid(city_dir):
    canopy_dir = city_dir+'/canopy'
    ground_dir = city_dir+'/ground'

    processing.run("LAStools:txt2lasPro", {'INPUT_GENERIC_DIRECTORY':canopy_dir,'INPUT_GENERIC_WILDCARDS':'*.txt','PARSE':'xyz','SKIP':0,'SCALE_FACTOR_XY':0.01,'SCALE_FACTOR_Z':0.01,'PROJECTION':0,'EPSG_CODE':4326,'UTM':0,'SP':0,'OUTPUT_DIRECTORY':canopy_dir,'OUTPUT_APPENDIX':'','OUTPUT_POINT_FORMAT':0,'ADDITIONAL_OPTIONS':'','CORES':4,'VERBOSE':False,'CPU64':False,'GUI':False})
    processing.run("LAStools:txt2lasPro", {'INPUT_GENERIC_DIRECTORY':ground_dir,'INPUT_GENERIC_WILDCARDS':'*.txt','PARSE':'xyz','SKIP':0,'SCALE_FACTOR_XY':0.01,'SCALE_FACTOR_Z':0.01,'PROJECTION':0,'EPSG_CODE':4326,'UTM':0,'SP':0,'OUTPUT_DIRECTORY':ground_dir,'OUTPUT_APPENDIX':'','OUTPUT_POINT_FORMAT':0,'ADDITIONAL_OPTIONS':'','CORES':4,'VERBOSE':False,'CPU64':False,'GUI':False})

    processing.run("LAStools:lasgridPro", {'INPUT_DIRECTORY':canopy_dir,'INPUT_WILDCARDS':'*.laz','MERGED':True,'FILTER_RETURN_CLASS_FLAGS1':0,'STEP':0.0002,'ATTRIBUTE':0,'METHOD':1,'USE_TILE_BB':False,'OUTPUT_DIRECTORY':city_dir,'OUTPUT_APPENDIX':'','OUTPUT_RASTER_FORMAT':0,'OUTPUT_RASTER':city_dir+'\\canopy.tif','ADDITIONAL_OPTIONS':'','CORES':4,'VERBOSE':False,'CPU64':False,'GUI':False})
    processing.run("LAStools:lasgridPro", {'INPUT_DIRECTORY':ground_dir,'INPUT_WILDCARDS':'*.laz','MERGED':True,'FILTER_RETURN_CLASS_FLAGS1':0,'STEP':0.0002,'ATTRIBUTE':0,'METHOD':0,'USE_TILE_BB':False,'OUTPUT_DIRECTORY':city_dir,'OUTPUT_APPENDIX':'','OUTPUT_RASTER_FORMAT':0,'OUTPUT_RASTER':city_dir+'\\ground.tif','ADDITIONAL_OPTIONS':'','CORES':4,'VERBOSE':False,'CPU64':False,'GUI':False})
    print('Gridded ICESat2 data')

def txt2grid(lastools_path, canopy_path, ground_path):
    las_text_canopy = lastools_path+'/txt2las -i '+ canopy_path+'/*.txt '+'-longlat -parse xyz'
    os.system(las_text_canopy)
    las_text_ground = lastools_path+'/txt2las -i '+ ground_path+'/*.txt '+'-longlat -parse xyz'
    os.system(las_text_ground)

    grid_text_canopy = lastools_path+'/lasgrid.exe -i '+ canopy_path+'/*.las '+'-longlat -highest -merged -o '+canopy_path+'/canopy.tif'+' -step 0.0002'
    os.system(grid_text_canopy)
    grid_text_ground = lastools_path+'/lasgrid.exe -i '+ ground_path+'/*.las '+'-longlat -highest -merged -o '+ground_path+'/ground.tif'+' -step 0.0002'
    os.system(grid_text_ground)
    print('Gridded ICESat2 data')

#import Canopy_ground
class canopy_Provider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(Canopy_ground.Canopy_ground())

    def id(self, *args, **kwargs):
        return 'Canopy_ground'

    def name(self, *args, **kwargs):
        return self.tr('Canopy_ground')

    def icon(self):
        return QgsProcessingProvider.icon(self)
    
#provider =canopy_Provider()
#QgsApplication.processingRegistry().addProvider(provider)

def icesat_ndsm(canopy_file,ground_file,output_path):
    alg_params = {
            'CELLSIZE': 0,
            'CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'EXPRESSION': '(("canopy@1"-"ground@1")>=0)*("canopy@1"-"ground@1")',
            'EXTENT': None,
            'LAYERS': [canopy_file,ground_file],
            'OUTPUT': output_path+'/icesat-2_ndsm.tif'
        }
    processing.run('qgis:rastercalculator', alg_params)
    
def icesat_gedi(txt_file,output_file):
    string = 'gdal_merge.py -a_nodata 0.0 -ot Float32 -of GTiff -o '+output_file+' --optfile '+txt_file
    #print(string)
    os.system(string)
    
def merge_gdal(txt_file,output_file):
    string = 'gdal_merge.py -ot Float32 -of GTiff -o '+output_file+' --optfile '+txt_file
    #print(string)
    os.system(string)
    
def to_pixels(merged_file,output_path):
    processing.run("native:pixelstopoints", {'INPUT_RASTER':merged_file,'RASTER_BAND':1,'FIELD_NAME':'VALUE','OUTPUT':output_path+'/vector.gpkg'})
    
def triangular(input_path,file,output_path):
    ext = QgsVectorLayer(input_path+'/'+str(file)).extent()
    processing.run("qgis:tininterpolation", {'INTERPOLATION_DATA':input_path+'/'+str(file)+'|layername='+os.path.splitext(str(file))[0]+'::~::0::~::1::~::2','METHOD':0,'EXTENT':str(ext.xMinimum())+','+str(ext.xMaximum())+','+str(ext.yMinimum())+','+str(ext.yMaximum())+' [EPSG:4326]','PIXEL_SIZE':0.00025,'OUTPUT':output_path+'/GEDI.tif'})
    
def space_ndsm(alos_file,gedi_file,output_path):  
    alg_params = {
            'CELLSIZE': 0,
            'CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'EXPRESSION': '(("ALOS@1"<=2)*"GEDI@1"*0.5)+(("ALOS@1">2)*ALOS@1)',
            'EXTENT': None,
            'LAYERS': [alos_file,gedi_file],
            'OUTPUT': output_path+'/space.tif'
        }
    processing.run('qgis:rastercalculator', alg_params)
    
def population(input_file,reference_file,output_path):
    ext = QgsRasterLayer(reference_file).extent()
    string = 'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -tr 0.000277 0.000277 -r cubicspline -te '+str(ext.xMinimum())+' '+str(ext.yMinimum())+' '+str(ext.xMaximum())+' '+str(ext.yMaximum())+' -te_srs EPSG:4326 -of GTiff '+input_file+' '+ output_path+'/pop.tif'
    os.system(string)

def fix_geometry(input_file,output_path,file_name):
    processing.run("native:fixgeometries", {'INPUT':input_file,'OUTPUT':output_path+'/'+file_name})

def merge_geomtery(input_ms,input_osm,output_path):
    processing.run("native:mergevectorlayers", {'LAYERS':[input_ms,input_osm+'|geometrytype=Polygon'],'CRS':QgsCoordinateReferenceSystem('EPSG:4326'),'OUTPUT':output_path+'/ms_osm.gpkg'})

def dissolve(input_file,output_path):
    processing.run("native:dissolve", {'INPUT':input_file+'|geometrytype=Polygon','FIELD':[],'OUTPUT':output_path+'/dissolved.gpkg'})
    
def single2multi(input_file,output_path):
    processing.run("native:multiparttosingleparts", {'INPUT':input_file,'OUTPUT':output_path+'/multi.gpkg'})

def delete_holes(input_file,output_path):
    processing.run("native:deleteholes", {'INPUT':input_file,'MIN_AREA':0,'OUTPUT':output_path+'/holes.gpkg'})

def zonal_pass1(vector_file, raster_file, output_path):
    processing.run("native:zonalstatisticsfb", {'INPUT':vector_file,'INPUT_RASTER':raster_file ,'RASTER_BAND':1,'COLUMN_PREFIX':'ALOS','STATISTICS':[6],'OUTPUT':output_path+'/Zonal_pass1.gpkg'})

def zonal_pass2(vector_file, raster_file, output_path):
    processing.run("native:zonalstatisticsfb", {'INPUT':vector_file,'INPUT_RASTER':raster_file ,'RASTER_BAND':1,'COLUMN_PREFIX':'Population','STATISTICS':[2],'OUTPUT':output_path+'/Zonal_pass2.gpkg'})

def zonal_pass3(vector_file,osm_file,output_path):
    processing.run("qgis:joinbylocationsummary", {'INPUT':vector_file,'JOIN':osm_file,'PREDICATE':[0,1,2,3,4,5,6],'JOIN_FIELDS':['OSM'],'SUMMARIES':[3],'DISCARD_NONMATCHING':False,'OUTPUT':output_path+'/osm_joined.gpkg'})

def merge(files_path,output_path,dataset):
    all_files = os.listdir(files_path)
    filenames = [file for file in all_files if file.endswith('.nc')]
    with open(files_path+'/merge.txt', 'w') as file:
        for filename in filenames:
            file.write(f"{files_path+'/'+filename}\n")
    merge_gdal(files_path+'/merge.txt',output_path+'/'+dataset+'.nc')
    
def merge2(files_path,output_path,dataset):
    all_files = os.listdir(files_path)
    filenames = [file for file in all_files if file.endswith('.nc')]
    with open(files_path+'/merge.txt', 'w') as file:
        for filename in filenames:
            file.write(f"{files_path+'/'+filename}\n")
    icesat_gedi(files_path+'/merge.txt',output_path+'/'+dataset+'.nc')
