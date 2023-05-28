#!/usr/bin/python3
# conda activate gee
# mpiexec -n 4 python3 Sentinel_2_Importer.py < country_name >

import ee
ee.Initialize()

import geemap
import os
import time
import sys
from datetime import date, timedelta
import pandas as pd
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

country_name = sys.argv[1].lower()
SITE = ' '.join([i.capitalize() for i in country_name.replace('_', ' ').split(' ')])

# Upcoming Parameters
CLOUD_FILTER = 90
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
STATE=False
GRIDSIZE=18000      #km*1000
DIR_PATH = f'/mnt/locutus/remotesensing/r62/modeling_agbd/{country_name}/Input'
SCALE=10
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
start_date = date(2019, 4, 20) # Year-Month-Day (minus 5 days to make it an even 38, 30 day intervals)
end_date = date(2022, 6, 2) # Year-Month-Day
INCREMENT = 30 # daily increment so in this case every 30 days. Check the variable 
# in this case titled date_ranges as the last month or two can sometimes have time frames 
# that are not 30 days. Changing start_date and end_date can fix this. 

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

###################################### Set Ranks by Number of Imported Bands #####################################

all_bands = np.array(BANDS)
all_bands = np.array_split(all_bands, size)
for b in range(len(all_bands)):
    if b == rank:
        bands_list = all_bands[b]
        # bs = ('_').join([word[1:] for word in all_bands[0]])
        DIR_PATH = f'{DIR_PATH}/Sentinel2/RANK_{b}'
    else:
        pass

print('RANK:', rank, bands_list, flush = True)
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

######################################## Sentinel-2 Functions Cloud Masking ##############################

# Create cloud and shadow mask definitions 
def get_s2_sr_cld_col(aoi, start_date, end_date):
    """Get Sentinel-2 data and join s2cloudless collection
    with Sentinel TOA data."""
    # Import and filter S2 TOA.

    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
        .sort('CLOUDY_PIXEL_PERCENTAGE', False))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def add_cloud_bands(img):
    """Create and add cloud bands."""
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    """Create and add cloud shadow bands."""
    # Identify water pixels from the SCL band.
    # not_water = img.select('SCL').neq(6) <- this is for SR, not TOA
    not_water = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    not_water = not_water.select('NDWI')

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    """Use cloud shadow and shadow bands to create masks."""
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    """Mask cloudy pixels."""
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


############################ Apply the Cloud Mask #########################################

s2_sr_cld_col = get_s2_sr_cld_col(grid, str(start_date), str(end_date))
s2_sr = (s2_sr_cld_col
         .map(add_cld_shdw_mask)
         .map(apply_cld_shdw_mask)).select(list(bands_list))#.map(replace_neg_Infs)

print("Shadow mask applied to tiles.")
start = time.time()

for RANGE in date_ranges:
    sentinel2_by_date = s2_sr.filterDate(RANGE[0], RANGE[1])
    for location, poly in enumerate(polys):
        sentinel2 = sentinel2_by_date.filterBounds(poly)
        meanImage = sentinel2.reduce(ee.Reducer.max())
        PATH = DIR_PATH + '/' + RANGE[0] + '_to_' + RANGE[1] + '/LOCATION_' + str(location) + '.tif'

        if os.path.isfile(PATH):
            print('FILE:', PATH, ' ALREADY EXISTS', flush=True)
        else:
            geemap.ee_export_image(ee.Image(meanImage.toFloat()),
                       filename=PATH, 
                        scale=10, # scale must be 10 to accomodate the 10m bands (weakest link)
                        region=poly, 
                        file_per_band=False)

stop = time.time()
time_taken = (stop - start)/3600
print('PROCESS COMPLETE! TOOK', time_taken, 'HOURS')
