import geopandas as gpd
import re
import pandas as pd
import os
import glob
import numpy as np
import shutil
from distutils.dir_util import copy_tree
from pyproj import Transformer
import subprocess
import time
from qgis.core import *
from qgis.gui import *
from qgis.PyQt.QtWidgets import *
import processing
from qgis.analysis import QgsNativeAlgorithms
from processing.core.Processing import Processing
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def urb_fra_index(path, dx, dy, lat, lon, tile_x, tile_y, urb_fra):
    file = open(path, "w")
    file.write('type=continuos'+ "\n")
    file.write('    projection=regular_ll'+ "\n")
    file.write('    missing_value=0.'+ "\n")
    file.write('    dx= '+str(dx)+ "\n")
    file.write('    dy= '+str(dy)+ "\n")
    file.write('    known_x=1.0'+ "\n")
    file.write('    known_y=1.0'+ "\n")
    file.write('    known_lat = '+str(lat)+ "\n")
    file.write('    known_lon = '+str(lon)+ "\n")
    file.write('    wordsize = 4'+ "\n")
    file.write('    tile_x= '+str(tile_x)+ "\n")
    file.write('    tile_y= '+str(tile_y)+ "\n")
    if urb_fra:
        file.write('    tile_z=1'+ "\n")
    else:
        file.write('    tile_z=132'+ "\n")
    file.write('    units="-"'+ "\n")
    file.write('    scale_factor=0.01'+ "\n")
    if urb_fra:
        file.write('    description="Urban fraction"'+ "\n")
    else:
        file.write('    description="GLOBUS morphology"'+ "\n")
    file.close()
    
def g2w(g2w_path, output_folder, globus_vector, esa_raster, city_name):

    print('Processing UCPs for '+globus_vector)
    source_path = globus_vector
    dest_path = output_folder+'/'
    esa_path = esa_raster
    layer = QgsVectorLayer(source_path)
    ext = layer.extent()
    crs = ' [EPSG 326'+layer.crs().description()[-3:][:2]+']'
    crs_ = 'EPSG:326'+layer.crs().description()[-3:][:2]
    processing.run("native:creategrid", {'TYPE':0,'EXTENT':str(ext.xMinimum())+','+str(ext.xMaximum())+','+str(ext.yMinimum())+','+str(ext.yMaximum())+crs,'HSPACING':300,'VSPACING':300,'HOVERLAY':0,'VOVERLAY':0,'CRS':QgsCoordinateReferenceSystem(crs_),'OUTPUT':dest_path+'point.gpkg'})
    print('Grid created')
    processing.run("native:buffer", {'INPUT':dest_path+'point.gpkg','DISTANCE':500,'SEGMENTS':4,'END_CAP_STYLE':2,'JOIN_STYLE':0,'MITER_LIMIT':2,'DISSOLVE':False,'OUTPUT':dest_path+'buffered.gpkg'})
    print('Grid buffered')
    processing.run("qgis:joinbylocationsummary", {'INPUT':dest_path+'buffered.gpkg|layername=buffered','JOIN':source_path,'PREDICATE':[1,5],'JOIN_FIELDS':['Area','Volume','Surface'],'SUMMARIES':[5],'DISCARD_NONMATCHING':False,'OUTPUT':dest_path+'others.gpkg'})
    print('UCPs joined')
    string = 'gdal_rasterize -a height -tr 2.0 2.0 -a_nodata 0.0 -te '+str(ext.xMinimum())+' '+str(ext.yMinimum())+' '+str(ext.xMaximum())+' '+str(ext.yMaximum())+' -ot Float32 -of GTiff '+source_path+' '+dest_path+'rasterized.tif'
    os.system(string)
    print('Heights rasterized')
    processing.run("native:zonalhistogram", {'INPUT_RASTER':dest_path+'rasterized.tif','RASTER_BAND':1,'INPUT_VECTOR':dest_path+'buffered.gpkg|layername=buffered','COLUMN_PREFIX':'HISTO_','OUTPUT':dest_path+'histo.gpkg'})
    print('Calculated histograms')
    os.remove(dest_path+'rasterized.tif')
    processing.run("native:zonalstatisticsfb", {'INPUT':dest_path+'buffered.gpkg|layername=buffered','INPUT_RASTER':esa_path ,'RASTER_BAND':1,'COLUMN_PREFIX':'ESA','STATISTICS':[2],'OUTPUT':dest_path+'ESA.gpkg'})
    print('Processed urban fraction')
    print('Getting all data togeather')
    point = gpd.read_file(dest_path+'point.gpkg')
    others = gpd.read_file(dest_path+'others.gpkg')
    histo = gpd.read_file(dest_path+'histo.gpkg')
    esa_urb = gpd.read_file(dest_path+'ESA.gpkg')
    lambda_p = (others.Area_sum.fillna(0)/(1000*1000)).tolist()
    lambda_b = (others.Surface_sum.fillna(0)/(1000*1000)).tolist()
    hgt = (others.Volume_sum/others.Area_sum).fillna(0).tolist()
    urb_fra = (esa_urb.ESAmean).fillna(0).tolist()
    arr = np.asarray(urb_fra)
    arr = arr.reshape(len(point.loc[point.top==point.top[0]]), len(point.loc[point.left==point.left[0]]))
    arr = np.fliplr(arr)
    urb_fra = arr.flatten().tolist()
    UTM_Y = point.top.tolist()
    UTM_X = point.left.tolist()
    lat = point.top.tolist()
    UTM_X = point.left.tolist()
    zone = layer.crs().description()[-3:][:2]
    trans = Transformer.from_crs("+proj=utm +zone="+zone+" +ellps=WGS84","epsg:4326",always_xy=True,)
    xx, yy = trans.transform(point["left"].values, point["top"].values)
    Lat = yy.tolist()
    Lon = xx.tolist()

    full = [f for f in histo.columns.tolist() if 'HISTO_' in f and has_numbers(f)]
    df_full = histo[full].sum(axis=1)
    h_5 = []
    h_10 = []
    h_15 = []
    h_20 = []
    h_25 = []
    h_30 = []
    h_35 = []
    h_40 = []
    h_45 = []
    h_50 = []
    h_55 = []
    h_60 = []
    h_65 = []
    h_70 = []
    h_75 = []

    for k in range(len(full)):
        if (int(re.sub('\\D', '', full[k]))>=1) and (int(re.sub('\\D', '', full[k]))<=6):
            h_5.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=7) and (int(re.sub('\\D', '', full[k]))<=10):
            h_10.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=11) and (int(re.sub('\\D', '', full[k]))<=15):
            h_15.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=16) and (int(re.sub('\\D', '', full[k]))<=20):
            h_20.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=21) and (int(re.sub('\\D', '', full[k]))<=25):
            h_25.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=26) and (int(re.sub('\\D', '', full[k]))<=30):
            h_30.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=31) and (int(re.sub('\\D', '', full[k]))<=35):
            h_35.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=36) and (int(re.sub('\\D', '', full[k]))<=40):
            h_40.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=41) and (int(re.sub('\\D', '', full[k]))<=45):
            h_45.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=46) and (int(re.sub('\\D', '', full[k]))<=50):
            h_50.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=51) and (int(re.sub('\\D', '', full[k]))<=55):
            h_55.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=56) and (int(re.sub('\\D', '', full[k]))<=60):
            h_60.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=61) and (int(re.sub('\\D', '', full[k]))<=65):
            h_65.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=65) and (int(re.sub('\\D', '', full[k]))<=70):
            h_70.append(full[k])
        elif (int(re.sub('\\D', '', full[k]))>=71):
            h_75.append(full[k])

    df_h5 = (histo[h_5].sum(axis=1)/df_full).fillna(0).tolist()
    df_h10 = (histo[h_10].sum(axis=1)/df_full).fillna(0).tolist()
    df_h15 = (histo[h_15].sum(axis=1)/df_full).fillna(0).tolist()
    df_h20 = (histo[h_20].sum(axis=1)/df_full).fillna(0).tolist()
    df_h25 = (histo[h_25].sum(axis=1)/df_full).fillna(0).tolist()
    df_h30 = (histo[h_30].sum(axis=1)/df_full).fillna(0).tolist()
    df_h35 = (histo[h_35].sum(axis=1)/df_full).fillna(0).tolist()
    df_h40 = (histo[h_40].sum(axis=1)/df_full).fillna(0).tolist()
    df_h45 = (histo[h_45].sum(axis=1)/df_full).fillna(0).tolist()
    df_h50 = (histo[h_50].sum(axis=1)/df_full).fillna(0).tolist()
    df_h55 = (histo[h_55].sum(axis=1)/df_full).fillna(0).tolist()
    df_h60 = (histo[h_60].sum(axis=1)/df_full).fillna(0).tolist()
    df_h65 = (histo[h_65].sum(axis=1)/df_full).fillna(0).tolist()
    df_h70 = (histo[h_70].sum(axis=1)/df_full).fillna(0).tolist()
    df_h75 = (histo[h_75].sum(axis=1)/df_full).fillna(0).tolist()

    dataframe = pd.DataFrame({'UTM_Y':UTM_Y,'UTM_X':UTM_X,'lat':Lat,'lon':Lon,'URB_FRA':urb_fra,'lambda_p':lambda_p,'lambda_b':lambda_b,'height':hgt,'H_5':df_h5,'H_10':df_h10,'H_15':df_h15,'H_20':df_h20,'H_25':df_h25,'H_30':df_h30
                             ,'H_35':df_h35,'H_40':df_h40,'H_45':df_h45,'H_50':df_h50,'H_55':df_h55,'H_60':df_h60,'H_65':df_h65,'H_70':df_h70,'H_75':df_h75})
    dataframe.loc[dataframe['lambda_p'] > dataframe['URB_FRA'], 'URB_FRA'] = dataframe['lambda_p']
    dataframe.to_csv(dest_path+city_name+'.csv',index=False)
    print('Writing text files')
    os.mkdir(dest_path+'others')
    dataframe['URB_FRA'].to_csv(dest_path+'others/'+'LambdaU_average_1kmx1km.txt', index=False,header=False)
    dataframe['lambda_p'].to_csv(dest_path+'others/'+'LambdaP_average_1kmx1km.txt', index=False,header=False)
    dataframe['lambda_b'].to_csv(dest_path+'others/'+'Lambdab_average_1kmx1km.txt', index=False,header=False)
    dataframe['height'].to_csv(dest_path+'others/'+'MeanBuildingHeight_average_1kmx1km.txt', index=False,header=False)
    os.mkdir(dest_path+'histograms')
    shutil.copy(g2w_path+'/histo_files.txt', dest_path+'histograms/histo_files')
    dataframe['H_5'].to_csv(dest_path+'histograms/H_05.txt', index=False,header=False)
    dataframe['H_10'].to_csv(dest_path+'histograms/H_10.txt', index=False,header=False)
    dataframe['H_15'].to_csv(dest_path+'histograms/H_15.txt', index=False,header=False)
    dataframe['H_20'].to_csv(dest_path+'histograms/H_20.txt', index=False,header=False)
    dataframe['H_25'].to_csv(dest_path+'histograms/H_25.txt', index=False,header=False)
    dataframe['H_30'].to_csv(dest_path+'histograms/H_30.txt', index=False,header=False)
    dataframe['H_35'].to_csv(dest_path+'histograms/H_35.txt', index=False,header=False)
    dataframe['H_40'].to_csv(dest_path+'histograms/H_40.txt', index=False,header=False)
    dataframe['H_45'].to_csv(dest_path+'histograms/H_45.txt', index=False,header=False)
    dataframe['H_50'].to_csv(dest_path+'histograms/H_50.txt', index=False,header=False)
    dataframe['H_55'].to_csv(dest_path+'histograms/H_55.txt', index=False,header=False)
    dataframe['H_60'].to_csv(dest_path+'histograms/H_60.txt', index=False,header=False)
    dataframe['H_65'].to_csv(dest_path+'histograms/H_65.txt', index=False,header=False)
    dataframe['H_70'].to_csv(dest_path+'histograms/H_70.txt', index=False,header=False)
    dataframe['H_75'].to_csv(dest_path+'histograms/H_75.txt', index=False,header=False)
    print('Done calculating UCPs for: '+city_name+"\n")

    urb_frc_path = dest_path+'/urb_fra/index'
    morph_path = dest_path+'/morph/index'
    tile_y = len(point.loc[point.left==point.left[0]])
    tile_x = len(point.loc[point.top==point.top[0]])
    yy_max= point["top"].iloc[0]
    yy_min = point["top"].iloc[-1]
    xx_min = point["left"].iloc[0]
    xx_max = point["left"].iloc[-1]

    copy_tree(g2w_path, dest_path)
    print('Processing binary files for '+city_name)
    f = open(dest_path+'/rd_wr_binary.f90', 'r')
    linelist = f.readlines()
    f.close
    f2 = open(dest_path+'/rd_wr_binary.f90', 'w')
    for line in linelist:
        line = line.replace('parameter (nx=212,ny=216,nz=1,nurbm=15,nzu=15)', 'parameter (nx='+str(tile_x)+',ny='+str(tile_y)+',nz=1,nurbm=15,nzu=15)')
        line = line.replace('iizone=31', 'iizone='+format(point.crs)[8:])
        line = line.replace('xx=421166.1701', 'xx='+str(xx_min))
        line = line.replace('yy=5372682.242', 'yy='+str(yy_min))
        line = line.replace('xx=484466.1701', 'xx='+str(xx_max))
        line = line.replace('yy=5437182.242', 'yy='+str(yy_max))

        f2.write(line)
    f2.close()

    os.chdir(dest_path)
    subprocess.run('make')
    time.sleep(5)
    subprocess.run('./rd_wr_binary.exe')

    df = pd.read_csv(dest_path+'/out.txt', header=None)
    df.columns = ['Values']
    lat = df['Values'][0][9:]
    lon = df['Values'][1][9:]
    dx = df['Values'][2][7:]
    dy = df['Values'][3][7:]

    urb_fra_index(urb_frc_path, dx, dy, lat, lon, tile_x, tile_y, urb_fra=True)
    urb_fra_index(morph_path, dx, dy, lat, lon, tile_x, tile_y, urb_fra=False)
    print('Done '+city_name)
