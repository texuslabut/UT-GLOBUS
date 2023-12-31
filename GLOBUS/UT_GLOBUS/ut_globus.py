import geopandas
import numpy as np
import requests, io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
import joblib
import utm
from geopy.geocoders import Nominatim
import pandas as pd
from shapely import wkt
import time

def ut_globus(input_file, output_path, city_name, rf_model):
    rf_random = joblib.load(rf_model)
    data = geopandas.read_file(input_file)
    if 'ESAmax' in data.columns:
        pass
    else:
        data["ESAmax"] = np.nan
    data = data[['geometry','ALOSmax','Populationmean','OSM_max']]
    #city_name = str(os.path.splitext(str(f))[0]).replace('_',' ')
    #print(city_name)
    geolocator = Nominatim(user_agent="my-app", timeout = 10)
    bounds = data.total_bounds
    center_point = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    center_point_str = f"{center_point[1]}, {center_point[0]}"
    location = geolocator.reverse(center_point_str)
    zone = '326' + str(utm.from_latlon(location.latitude, location.longitude)[2])
    #location = geolocator.geocode(city_name)
    #zone = '326'+str(utm.from_latlon(location.latitude, location.longitude)[2])
    data = data.to_crs("epsg:"+zone)
    data['Area'] = data.area
    data['Perimeter'] = data.length
    if data.empty:
        pass
    else:
        validation_ip = data[['ALOSmax','Populationmean','Perimeter','Area']]
        validation_ip['ALOSmax'].fillna(3, inplace=True)
        validation_ip['Populationmean'].fillna(500, inplace=True)
        validation_ip.replace([np.inf, -np.inf], 3, inplace=True)
        validation_op = rf_random.predict(validation_ip)
        data['Height'] = validation_op
        #data.ESAmax.fillna(data.OSM_max, inplace=True)
        data.OSM_max.fillna(data.Height, inplace=True)
        data.rename(columns={'OSM_max': 'height'}, inplace=True)
        data[data['height']>=500] == 4
        data[data['height']<=-1] == 4
    
        data['Volume'] = data.Height*data.Area
        data['Surface'] = (data.Perimeter*data.Height)+data.Area
        data = data[['geometry','height','Area', 'Volume', 'Surface']]
        data = data.round({'height': 0,'Area':0, 'Volume':0, 'Surface':0})
        data.to_file(output_path+'/'+city_name+'.gpkg', driver='GPKG', layer='GLOBUS')
