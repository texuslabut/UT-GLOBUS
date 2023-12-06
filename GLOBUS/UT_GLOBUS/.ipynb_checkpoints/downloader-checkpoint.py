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


def Microsoft_Footprints(X_min, Y_min,  X_max, Y_max, path2, city_name):
    
    df = pd.read_csv("https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv")
    
    aoi_geom = {"coordinates": [ [
        [X_min, Y_max],
        [X_min, Y_min],
        [X_max, Y_min],
        [X_max, Y_max],
        [X_min, Y_max],
        ]
],
         
"type": "Polygon",
               }
    try:
        aoi_shape = shapely.geometry.shape(aoi_geom)
        minx, miny, maxx, maxy = aoi_shape.bounds

        output_fn = path2+'/'+city_name+'_MS'.replace(" ", "_")+'.gpkg'

        quad_keys = set()
        for tile in list(mercantile.tiles(minx, miny, maxx, maxy, zooms=9)):
            quad_keys.add(int(mercantile.quadkey(tile)))
        quad_keys = list(quad_keys)
        print(f"The input area spans {len(quad_keys)} tiles: {quad_keys}")

        idx = 0
        combined_rows = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_fns = []
            for quad_key in tqdm(quad_keys):
                rows = (df[df["QuadKey"] == quad_key])
                rows = rows.sort_values(by=['Size']).head(1)
                try:
                    if rows.shape[0] == 1:
                        url = rows.iloc[0]["Url"]

                        df2 = pd.read_json(url, lines=True)
                        df2["geometry"] = df2["geometry"].apply(shapely.geometry.shape)

                        gdf = gpd.GeoDataFrame(df2, crs=4326)
                        fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                        tmp_fns.append(fn)
                        if not os.path.exists(fn):
                            gdf.to_file(fn, driver="GeoJSON")

                    elif rows.shape[0] > 1:
                        raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
                    else:
                        raise ValueError(f"QuadKey not found in dataset: {quad_key}")
                except:
                    None

            for fn in tmp_fns:
                with fiona.open(fn, "r") as f:
                    for row in tqdm(f):
                        row = dict(row)
                        shape = shapely.geometry.shape(row["geometry"])

                        if aoi_shape.contains(shape):
                            if "id" in row:
                                del row["id"]
                            row["properties"] = {"id": idx}
                            idx += 1
                            combined_rows.append(row)

        schema = {"geometry": "Polygon", "properties": {"id": "int"}}

        with fiona.open(output_fn, "w", driver="GPKG", schema=schema) as f:
            f.writerecords(combined_rows)
            tempfile.TemporaryDirectory().cleanup()
    except:
        print('Skipping '+city_name)

def icesat_2(X_min, Y_min, X_max, Y_max, path, city_name, username, email_id):
    print('Processing ICESat-2: '+city_name)
    os.mkdir(f"{path}/icesat2_{city_name}")
    os.mkdir(f"{path}/icesat2_{city_name}/ATL03")
    os.mkdir(f"{path}/icesat2_{city_name}/ATL08")

    region_03 = ipx.Query('ATL03',[X_min, Y_min, X_max, Y_max],['2018-11-01','2019-10-31'])
    region_03.earthdata_login(username, email_id)
    region_03.order_vars.append(var_list=["ds_surf_type","ds_xyz","atlas_sdp_gps_epoch","control","data_end_utc","data_start_utc","end_cycle","end_delta_time","end_geoseg","end_gpssow","end_gpsweek","end_orbit","end_region","end_rgt","granule_end_utc","granule_start_utc","release","start_cycle","start_delta_time","start_geoseg","start_gpssow","start_gpsweek","start_orbit","start_region","start_rgt","version","atl03_pad","band_tol","min_full_sat","min_near_sat","min_sat_h","min_scan_s","ph_sat_flag","ph_sat_lb","ph_sat_ub","podppd_pad","scan_settle_s","det_ab_flag","ds_gt","ds_stat","hvpc_ab_flag","laser_12_flag","lrs_ab_flag","pdu_ab_flag","ph_uncorrelated_error","spd_ab_flag","tams_ab_flag","rx_bckgrd_sensitivity","rx_return_sensitivity","tx_pulse_distribution","tx_pulse_energy","tx_pulse_skew_est","tx_pulse_thresh_lower","tx_pulse_thresh_upper","tx_pulse_width_lower","tx_pulse_width_upper","ds_channel","cal42_product","side","temperature","gt1l","gt1r","gt2l","gt2r","gt3l","gt3r","cal34_product","cal19_product","bin_width","cal20_product","hist_x","laser","mode","num_bins","return_source","addpad_flag","alpha_inc","alpha_max","delta_t_gap_min","delta_t_lin_fit","delta_t_max","delta_t_min","delta_z_bg","delta_zmax2","delta_zmin","e_a","e_linfit_edit","e_linfit_slant","e_m","e_m_mult","htspanmin","lslant_flag","min_fit_time_fact","nbin_min","n_delta_z1","n_delta_z2","nphot_min","nslw","nslw_v","out_edit_flag","pc_bckgrd_flag","r","r2","sig_find_t_inc","snrlow","snrmed","t_gap_big","min_tep_ph","min_tep_secs","n_tep_bins","tep_bin_size","tep_gap_size","tep_normalize","tep_peak_bins","tep_prim_window","tep_range_prim","tep_rm_noise","tep_sec_window","tep_start_x","tep_valid_spot","reference_tep_flag","tep_bckgrd","tep_duration","tep_hist","tep_hist_sum","tep_hist_time","tep_tod","bckgrd_counts","bckgrd_counts_reduced","bckgrd_hist_top","bckgrd_int_height","bckgrd_int_height_reduced","bckgrd_rate","delta_time","pce_mframe_cnt","tlm_height_band1","tlm_height_band2","tlm_top_band1","tlm_top_band2","altitude_sc","bounce_time_offset","full_sat_fract","near_sat_fract","neutat_delay_derivative","neutat_delay_total","neutat_ht","ph_index_beg","pitch","podppd_flag","range_bias_corr","ref_azimuth","ref_elev","reference_photon_index","reference_photon_lat","reference_photon_lon","roll","segment_dist_x","segment_id","segment_length","segment_ph_cnt","sigma_across","sigma_along","sigma_h","sigma_lat","sigma_lon","solar_azimuth","solar_elevation","surf_type","velocity_sc","yaw","dac","dem_flag","dem_h","geoid","geoid_free2mean","tide_earth","tide_earth_free2mean","tide_equilibrium","tide_load","tide_ocean","tide_oc_pole","tide_pole","dist_ph_across","dist_ph_along","h_ph","lat_ph","lon_ph","ph_id_channel","ph_id_count","ph_id_pulse","quality_ph","signal_conf_ph","bckgrd_mean","bckgrd_sigma","t_pc_delta","z_pc_delta","crossing_time","cycle_number","lan","orbit_number","rgt","sc_orient","sc_orient_time","qa_granule_fail_reason","qa_granule_pass_fail","qa_perc_signal_conf_ph_high","qa_perc_signal_conf_ph_low","qa_perc_signal_conf_ph_med","qa_perc_surf_type","qa_total_signal_conf_ph_high","qa_total_signal_conf_ph_low","qa_total_signal_conf_ph_med"])
    region_03.subsetparams(Coverage=region_03.order_vars.wanted)
    region_03.order_granules()
    region_03.download_granules(f"{path}/icesat2_{city_name}/ATL03")

    region_08 = ipx.Query('ATL08',[X_min, Y_min, X_max, Y_max],['2018-11-01','2019-10-31'])
    region_08.earthdata_login(username, email_id)
    region_08.order_vars.append(var_list=["ds_geosegments","ds_metrics","ds_surf_type","atlas_sdp_gps_epoch","control","data_end_utc","data_start_utc","end_cycle","end_delta_time","end_geoseg","end_gpssow","end_gpsweek","end_orbit","end_region","end_rgt","granule_end_utc","granule_start_utc","qa_at_interval","release","start_cycle","start_delta_time","start_geoseg","start_gpssow","start_gpsweek","start_orbit","start_region","start_rgt","version","atl08_region","bin_size_h","bin_size_n","bright_thresh","ca_class","can_noise_thresh","canopy20m_thresh","canopy_flag_switch","canopy_seg","can_stat_thresh","class_thresh","cloud_filter_switch","del_amp","del_mu","del_sigma","dem_filter_switch","dem_removal_percent_limit","dragann_switch","dseg","dseg_buf","fnlgnd_filter_switch","gnd_stat_thresh","gthresh_factor","h_canopy_perc","iter_gnd","iter_max","lseg","lseg_buf","lw_filt_bnd","lw_gnd_bnd","lw_toc_bnd","lw_toc_cut","max_atl03files","max_atl09files","max_peaks","max_try","min_nphs","n_dec_mode","night_thresh","noise_class","outlier_filter_switch","ph_removal_percent_limit","proc_geoseg","psf","p_static","ref_dem_limit","ref_finalground_limit","relief_hbot","relief_htop","shp_param","sig_rsq_search","sseg","stat20m_thresh","stat_thresh","tc_thresh","te_class","terrain20m_thresh","toc_class","up_filt_bnd","up_gnd_bnd","up_toc_bnd","up_toc_cut","asr","atlas_pa","beam_azimuth","beam_coelev","brightness_flag","cloud_flag_atm","cloud_fold_flag","delta_time","delta_time_beg","delta_time_end","dem_flag","dem_h","dem_removal_flag","h_dif_ref","last_seg_extend","latitude","latitude_20m","layer_flag","longitude","longitude_20m","msw_flag","night_flag","n_seg_ph","ph_ndx_beg","ph_removal_flag","psf_flag","rgt","sat_flag","segment_id_beg","segment_id_end","segment_landcover","segment_snowcover","segment_watermask","sigma_across","sigma_along","sigma_atlas_land","sigma_h","sigma_topo","snr","solar_azimuth","solar_elevation","surf_type","terrain_flg","urban_flag","canopy_h_metrics","canopy_h_metrics_abs","canopy_openness","canopy_rh_conf","centroid_height","h_canopy","h_canopy_20m","h_canopy_abs","h_canopy_quad","h_canopy_uncertainty","h_dif_canopy","h_max_canopy","h_max_canopy_abs","h_mean_canopy","h_mean_canopy_abs","h_median_canopy","h_median_canopy_abs","h_min_canopy","h_min_canopy_abs","n_ca_photons","n_toc_photons","photon_rate_can","segment_cover","subset_can_flag","toc_roughness","h_te_best_fit","h_te_best_fit_20m","h_te_interp","h_te_max","h_te_mean","h_te_median","h_te_min","h_te_mode","h_te_rh25","h_te_skew","h_te_std","h_te_uncertainty","n_te_photons","photon_rate_te","subset_te_flag","terrain_slope","classed_pc_flag","classed_pc_indx","d_flag","ph_h","ph_segment_id","crossing_time","cycle_number","lan","orbit_number","sc_orient","sc_orient_time","qa_granule_fail_reason","qa_granule_pass_fail"])
    region_08.subsetparams(Coverage=region_08.order_vars.wanted)
    region_08.order_granules()
    region_08.download_granules(f"{path}/icesat2_{city_name}/ATL08")

def qualityMask(im):
    return im.updateMask(im.select('quality_flag').eq(1)).select('rh100').toInt()

def gedi(X_min, Y_min,  X_max, Y_max, path2, city_name):
    print('Processing GEDI '+ city_name)
    if os.path.isdir(path2+'/temp'):
        pass
    else:
        os.mkdir(f"{path2}/temp")

    if ((X_max-X_min <=np.abs(0.25)) and (Y_max-Y_min<=np.abs(0.25))):
        aoi = ee.Geometry.Polygon(
    [[[X_min, Y_max],
      [X_min, Y_min],
      [X_max, Y_min],
      [X_max, Y_max]]])
        gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').map(qualityMask).filterBounds(aoi)
        ds = gedi.wx.to_xarray(region=aoi.bounds(), scale=25)
        data = ds.rh100.mean(dim='time', skipna=True)
        data.rio.write_crs('epsg:4326', inplace=True)
        data.to_netcdf(path2+ "/" +city_name+"_GEDI.nc")
        print('Done  GEDI'+ city_name)


    else:
        step_x =  (X_max-X_min)/5
        step_y =  (Y_max-Y_min)/5
        x_min = X_min
        y_max = Y_max

        for tile_y in range(1,6):
            for tile_x in range(1,6):
                aoi = ee.Geometry.Polygon(
    [[[x_min, y_max],
      [x_min, y_max-(step_y)],
      [x_min+(step_x), y_max-(step_y)],
      [x_min+(step_x), y_max]]])
                gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').map(qualityMask).filterBounds(aoi)
                ds = gedi.wx.to_xarray(region=aoi.bounds(), scale=25)
                data = ds.rh100.mean(dim='time', skipna=True)
                data.rio.write_crs('epsg:4326', inplace=True)
                data.to_netcdf(path2 + "/temp/"+city_name+'_'+str(tile_y)+'_'+str(tile_x)+".nc")
                x_min = x_min+(step_x)
            y_max = y_max-(step_y)
            x_min = X_min

        ds = xarray.merge([xarray.open_dataset(f) for f in glob.glob(path2 + "/temp/*.nc")])
        ds.to_netcdf(path2+ "/" +city_name+"_GEDI.nc")
        delete = glob.glob(path2 + "/temp/*.nc")
        for zz in delete:
            os.remove(zz)

def esa(X_min, Y_min,  X_max, Y_max, path2, city_name):
    
    print('Processing ESA '+ city_name)
    if os.path.isdir(path2+'/temp'):
        pass
    else:
        os.mkdir(f"{path2}/temp")

    if ((X_max-X_min <=np.abs(0.25)) and (Y_max-Y_min<=np.abs(0.25))):
        aoi = ee.Geometry.Polygon(
[[[X_min, Y_max],
  [X_min, Y_min],
  [X_max, Y_min],
  [X_max, Y_max]]])
        dataset_esa =ee.ImageCollection("ESA/WorldCover/v100").filterBounds(aoi)
        ds_esa = dataset_esa.wx.to_xarray(region=aoi.bounds(), scale=10)
        x_ = ds_esa.Map.values
        x_[x_!=50] = 0
        x_[x_==50] = 1
        ds_esa['Map'] = (('y', 'x'), x_[0,:,:])
        ds_esa.rio.write_crs('epsg:4326', inplace=True)
        ds_esa.to_netcdf(path2+'/'+city_name+"_ESA.nc")
        print('Done  ESA'+city_name)


    else:
        step_x =  (X_max-X_min)/2
        step_y =  (Y_max-Y_min)/2
        x_min = X_min
        y_max = Y_max


        for tile_y in range(1,3):
            for tile_x in range(1,3):
                aoi = ee.Geometry.Polygon(
[[[x_min, y_max],
  [x_min, y_max-(step_y)],
  [x_min+(step_x), y_max-(step_y)],
  [x_min+(step_x), y_max]]])
                dataset_esa =ee.ImageCollection("ESA/WorldCover/v100").filterBounds(aoi)
                ds_esa = dataset_esa.wx.to_xarray(region=aoi.bounds(), scale=10)
                x_ = ds_esa.Map.values
                x_[x_!=50] = 0
                x_[x_==50] = 1
                ds_esa['Map'] = (('y', 'x'), x_[0,:,:])
                ds_esa.rio.write_crs('epsg:4326', inplace=True)
                ds_esa.to_netcdf(path2 + "/temp/"+city_name+'_'+str(tile_y)+'_'+str(tile_x)+".nc")
                x_min = x_min+(step_x)
            y_max = y_max-(step_y)
            x_min = X_min

        ds = xarray.merge([xarray.open_dataset(f) for f in glob.glob(path2 + "/temp/*.nc")])
        ds.to_netcdf(path2+'/'+city_name+"_ESA.nc")
        delete = glob.glob(path2 + "/temp/*.nc")
        for zz in delete:
            os.remove(zz)

def renameField(srcLayer, oldFieldName, newFieldName):
    ds = gdal.OpenEx(srcLayer.source(), gdal.OF_VECTOR | gdal.OF_UPDATE)
    ds.ExecuteSQL('ALTER TABLE {} RENAME COLUMN {} TO {}'.format(srcLayer.name(), oldFieldName, newFieldName))
    srcLayer.reload()

def osm(X_min, Y_min, X_max, Y_max, path2, osm_path, city_name):    
    print('Processing OSM: '+ city_name)
    urlStr = 'http://overpass-api.de/api/map?bbox='+str(X_min)+','+str(Y_min)+','+str(X_max)+','+str(Y_max)
    with urllib.request.urlopen(urlStr) as response:
        osmXml = response.read()
        osmXml = osmXml.decode('UTF-8')
    osmPath = path2 + "/OSM_building_" + city_name+'.osm'
    osmFile = open(osmPath, 'w', encoding='utf-8')
    osmFile.write(osmXml)
    if os.fstat(osmFile.fileno()).st_size < 1:
        urlStr = 'http://api.openstreetmap.org/api/0.6/map?bbox='+str(X_min)+','+str(Y_min)+','+str(X_max)+','+str(Y_max)
        with urllib.request.urlopen(urlStr) as response:
            osmXml = response.read()
            osmXml = osmXml.decode('UTF-8')
        osmPath = path2 + "/OSM_building_" + city_name+'.osm'
        osmFile = open(osmPath, 'w', encoding='utf-8')
        osmFile.write(osmXml)

    osmFile.close()

    osmconf_dir = osm_path+'/osmconf.ini'
    gdal.SetConfigOption("OSM_CONFIG_FILE", osmconf_dir)
    osm_option = gdal.VectorTranslateOptions(options=[
    '-skipfailures', 
    '-t_srs', 'EPSG:4326' ,
    '-overwrite',
    '-nlt', 'MULTIPOLYGON',
    '-f', 'ESRI Shapefile'])
            
    parent_dir = path2
    dir = str(city_name)
    path = os.path.join(parent_dir, dir)
    os.mkdir(path)
    outputshp = path+'/'
    gdal.VectorTranslate(outputshp,path2 + "/OSM_building_" + city_name+'.osm' , options=osm_option)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    driver.DeleteDataSource(outputshp + 'lines.shp')
    driver.DeleteDataSource(outputshp + 'multilinestrings.shp')
    driver.DeleteDataSource(outputshp + 'other_relations.shp')
    driver.DeleteDataSource(outputshp + 'points.shp')
    osmPolygonPath = outputshp + 'multipolygons.shp'
    vlayer = QgsVectorLayer(osmPolygonPath, 'multipolygons', 'ogr') # Reads temp file made from OSM data
    polygon_layer = vlayer
    fileInfo = QFileInfo(polygon_layer.source())
    polygon_ln = fileInfo.baseName()
#
    vlayer.startEditing()
    renameField(vlayer, 'building_l', 'bld_levels')
    renameField(vlayer, 'building_h', 'bld_hght')
    renameField(vlayer, 'building_c', 'bld_colour')
    renameField(vlayer, 'building_m', 'bld_materi')
    renameField(vlayer, 'building_u', 'bld_use')
    vlayer.commitChanges()
    print('Done OSM: '+city_name)
    os.remove(path2 + "/OSM_building_" + city_name+'.osm' )


def alos(X_min, Y_min,  X_max, Y_max, path2, city_name):
    print('Processing ALOS '+ city_name)
    if os.path.isdir(path2+'/temp'):
        pass
    else:
        os.mkdir(f"{path2}/temp")

    if ((X_max-X_min <=np.abs(0.5)) and (Y_max-Y_min<=np.abs(0.5))):
        aoi = ee.Geometry.Polygon(
    [[[X_min, Y_max],
      [X_min, Y_min],
      [X_max, Y_min],
      [X_max, Y_max]]])
        dataset_alos = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM').filterBounds(aoi)
        dataset_srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').set('system:time_start', 0).clip(aoi)
        ds_alos = dataset_alos.wx.to_xarray(region=aoi.bounds(), scale=30)
        ds_srtm = dataset_srtm.wx.to_xarray(region=aoi.bounds(), scale=30)
        x = ds_alos.DSM.values[0,:,:]-ds_srtm.elevation.values[0,:,:]
        x[x<0]=0
        ds_alos['DSM'] = (('y', 'x'), x)
        ds_alos.rio.write_crs('epsg:4326', inplace=True)
        ds_alos.to_netcdf(path2+'/'+city_name+"_ALOS.nc")
        print('Done ALOS '+city_name)

    else:
        step_x =  (X_max-X_min)/2
        step_y =  (Y_max-Y_min)/2
        x_min = X_min
        y_max = Y_max


        for tile_y in range(1,3):
            for tile_x in range(1,3):
                aoi = ee.Geometry.Polygon(
    [[[x_min, y_max],
      [x_min, y_max-(step_y)],
      [x_min+(step_x), y_max-(step_y)],
      [x_min+(step_x), y_max]]])
                print('Alos')
                dataset_alos = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM').filterBounds(aoi)
                dataset_srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').set('system:time_start', 0).clip(aoi)
                ds_alos = dataset_alos.wx.to_xarray(region=aoi.bounds(), scale=30)
                ds_srtm = dataset_srtm.wx.to_xarray(region=aoi.bounds(), scale=30)
                x = ds_alos.DSM.values[0,:,:]-ds_srtm.elevation.values[0,:,:]
                x[x<0]=0
                ds_alos['DSM'] = (('y', 'x'), x)
                ds_alos.rio.write_crs('epsg:4326', inplace=True)
                ds_alos.to_netcdf(path2 + "/temp/"+city_name+'_'+str(tile_y)+'_'+str(tile_x)+".nc")
                x_min = x_min+(step_x)
            y_max = y_max-(step_y)
            x_min = X_min[k]
        ds = xarray.merge([xarray.open_dataset(f) for f in glob.glob(path2 + "/temp/*.nc")])
        ds.to_netcdf(path2+'/'+city_name+"_ALOS.nc")
        delete = glob.glob(path2 + "/temp/*.nc")
        for zz in delete:
            os.remove(zz)

def population(X_min, Y_min,  X_max, Y_max, path2, city_name):
    
    print('Processing Landscan population '+ city_name)
    if os.path.isdir(path2+'/temp'):
        pass
    else:
        os.mkdir(f"{path2}/temp")

    if ((X_max-X_min <=np.abs(1.0)) and (Y_max-Y_min<=np.abs(1.0))):
        aoi = ee.Geometry.Polygon(
[[[X_min, Y_max],
  [X_min, Y_min],
  [X_max, Y_min],
  [X_max, Y_max]]])
        dataset_pop =ee.ImageCollection("projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL").filterBounds(aoi)
        ds_pop = dataset_pop.wx.to_xarray(region=aoi.bounds(), scale=1000)
        
        x_ = ds_pop.b1.values
        ds_pop['b1'] = (('y', 'x'), x_[22,:,:])
        
        data = ds_pop
        data.rio.write_crs('epsg:4326', inplace=True)
        data.to_netcdf(path2+'/'+city_name+"_Pop.nc")
        print('Done  Population'+city_name)


    else:
        step_x =  (X_max-X_min)/2
        step_y =  (Y_max-Y_min)/2
        x_min = X_min
        y_max = Y_max


        for tile_y in range(1,3):
            for tile_x in range(1,3):
                aoi = ee.Geometry.Polygon(
[[[x_min, y_max],
  [x_min, y_max-(step_y)],
  [x_min+(step_x), y_max-(step_y)],
  [x_min+(step_x), y_max]]])
                dataset_pop =ee.ImageCollection("projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL").filterBounds(aoi)
                ds_pop = dataset_pop.wx.to_xarray(region=aoi.bounds(), scale=1000)
                
                x_ = ds_pop.b1.values
                ds_pop['b1'] = (('y', 'x'), x_[22,:,:])
                
                data = ds_pop
                data.rio.write_crs('epsg:4326', inplace=True)
                data.to_netcdf(path2 + "/temp/"+city_name+'_'+str(tile_y)+'_'+str(tile_x)+".nc")
                x_min = x_min+(step_x)
            y_max = y_max-(step_y)
            x_min = X_min

        ds = xarray.merge([xarray.open_dataset(f) for f in glob.glob(path2 + "/temp/*.nc")])
        ds.to_netcdf(path2+'/'+city_name+"_Pop.nc")
        delete = glob.glob(path2 + "/temp/*.nc")
        for zz in delete:
            os.remove(zz)