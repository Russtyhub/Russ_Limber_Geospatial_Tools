#!/usr/bin/python3
# conda activate gee
# mpiexec -n 2 python3 S1_Importer.py < country_name >

import ee
ee.Initialize()

import geemap
import os
import time
import math
import sys
from datetime import date, timedelta
import pandas as pd
import numpy as np
import datetime 
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

country_name = sys.argv[1].lower()
SITE = ' '.join([i.capitalize() for i in country_name.replace('_', ' ').split(' ')])
STATE = False
GRIDSIZE = 18000
DIR_PATH = f'/mnt/locutus/remotesensing/r62/modeling_agbd/{country_name}/Input'
start_date = date(2019, 4, 20) # Year-Month-Day (minus 5 days to make it an even 38, 30 day intervals)
end_date = date(2022, 6, 2) # Year-Month-Day
INCREMENT = 30 # daily increment so in this case every 30 days

####################################### Set Date Ranges #######################################

def create_list_of_dates(start_date, end_date):
    dates = []
    delta = end_date - start_date   # returns timedelta

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        dates.append(day)
    return dates

def create_time_intervals(dates_list, Interval):
    time_df = pd.DataFrame({'Date': dates_list}).astype('datetime64[ns]')
    interval = timedelta(Interval)
    grouped_cr = time_df.groupby(pd.Grouper(key='Date', freq=interval))
    date_ranges = []
    for i in grouped_cr:
        date_ranges.append(((str(i[1].min()[0]).split(' ')[0]), (str(i[1].max()[0]).split(' ')[0])))
    return date_ranges

date_ranges = create_time_intervals(create_list_of_dates(start_date, end_date), INCREMENT)

###################################### Create directories by Rank to add files to later #####################################

for b in range(size):
    if b == rank:
        DIR_PATH = f'/mnt/locutus/remotesensing/r62/modeling_agbd/{country_name}/Input/Sentinel1/RANK_{b}'
    else:
        pass

for RANGE in date_ranges:
    if os.path.isdir(DIR_PATH + '/' + RANGE[0] + '_to_' + RANGE[1]):
        pass
    else:
        os.makedirs(DIR_PATH + '/' + RANGE[0] + '_to_' + RANGE[1])

###################################### Set Location ###############################################################

# Import admin data and select country to create grid around
if STATE:
    br = (ee.FeatureCollection("FAO/GAUL/2015/level1")
            .filterMetadata('ADM0_NAME', 'equals', SITE)
            .filterMetadata('ADM1_NAME', 'equals', STATE)
      )
    print("Feature collection of {}, {}, loaded.".format(STATE, SITE))

else:
    br = (ee.FeatureCollection("FAO/GAUL/2015/level1")
            .filterMetadata('ADM0_NAME', 'equals', SITE))
    print("Feature collection of {} loaded.".format(SITE))

######################################## Create Grid and Polygons #########################################

# Create grid
# https://developers.google.com/earth-engine/tutorials/community/drawing-tools

def make_grid(region, a_scale):
    """
    Creates a grid around a specified ROI.
    User inputs their reasonably small ROI.
    User inputs a scale where 100000 = 100km.
    """
    # Creates image with 2 bands ('longitude', 'latitude') in degrees
    lonLat = ee.Image.pixelLonLat()

    # Select bands, multiply times big number, and truncate
    lonGrid = (lonLat
               .select('latitude')
               .multiply(10000000)
               .toInt()
              )
    latGrid = (lonLat
              .select('longitude')
              .multiply(10000000)
              .toInt()
              )

    # Multiply lat and lon images and reduce to vectors
    grid = (lonGrid
            .multiply(latGrid)
            .reduceToVectors(
                geometry = region,
                scale = a_scale, # 100km-sized boxes needs 100,000
                geometryType = 'polygon')
           )
    
    return(grid)

# Make your grid superimposed over ROI and limit tiles to 100
grid_xkm = make_grid(br, GRIDSIZE)

# Create dictionary of grid coordinates
grid_dict = grid_xkm.getInfo()
feats = grid_dict['features']

# Create a list of several ee.Geometry.Polygons
polys = []
for d in feats:
    coords = d['geometry']['coordinates']
    poly = ee.Geometry.Polygon(coords)
    polys.append(poly)

print("{} squares created.".format(len(polys)), flush=True)

# Might need to make grid smaller if it's huge

# Make the whole grid a feature collection for export purposes
grid = ee.FeatureCollection(polys)
num_km = GRIDSIZE / 1000
print("{} km grids created within {}.".format(num_km, country_name.title()), flush=True)

############################ Download Sentinel-1 Image Collection #########################################

sentinel1 =  ee.ImageCollection('COPERNICUS/S1_GRD')#.map(apply_rtf)
sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('instrumentMode', 'IW')) 

# Separate ascending and descending orbit images into distinct collections.
asc = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
desc = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

############################ Download Sentinel-1 Image by Time and Location #########################################

start = time.time()

for RANGE in date_ranges:
    if rank == 0:
        for location, poly in enumerate(polys):
            asc_by_date = asc.filterDate(RANGE[0], RANGE[1])
            composite = ee.Image.cat([asc_by_date.select('VH').mean(), asc_by_date.select('VV').mean(),]).focal_median()
            PATH = DIR_PATH + '/' + RANGE[0] + '_to_' + RANGE[1] + '/ASC_LOCATION_' + str(location) + '.tif'

            if os.path.isfile(PATH):
                print('FILE:', PATH, ' ALREADY EXISTS', flush=True)
            else:
                geemap.ee_export_image(ee.Image(composite.toFloat()),
                           filename=PATH, 
                            scale=10, # scale must be 10 to accomodate the 10m bands (weakest link)
                            region=poly, 
                            file_per_band=False)
    elif rank == 1:
        for location, poly in enumerate(polys):
            desc_by_date = desc.filterDate(RANGE[0], RANGE[1])
            composite = ee.Image.cat([desc_by_date.select('VH').mean(), desc_by_date.select('VV').mean(),]).focal_median()
            PATH = DIR_PATH + '/' + RANGE[0] + '_to_' + RANGE[1] + '/DESC_LOCATION_' + str(location) + '.tif'

            if os.path.isfile(PATH):
                print('FILE:', PATH, ' ALREADY EXISTS', flush=True)
            else:
                geemap.ee_export_image(ee.Image(composite.toFloat()),
                           filename=PATH, 
                            scale=10, # scale must be 10 to accomodate the 10m bands (weakest link)
                            region=poly, 
                            file_per_band=False)
				
stop = time.time()
time_taken = (stop - start)/3600
print('PROCESS COMPLETE! TOOK', time_taken, 'HOURS')
