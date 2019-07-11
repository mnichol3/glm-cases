"""
Author: Matt Nicholson

Functions to open and manipulate GOES-16 GLM netCDF files
"""
from os.path import isfile
from netCDF4 import Dataset
import pyproj
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sys import exit
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.cm as cm
from math import degrees, radians, atan, cos, sqrt

from grib import trunc
from proj_utils import scan_to_geod, geod_to_scan
from localglmfile import LocalGLMFile



def trim_header(abs_path):
    if (not isfile(abs_path)):
        raise OSError('File does not exist:', abs_path)

    if (not isfile(abs_path + '.nc') and abs_path[-3:] != '.nc'):

        with open(abs_path, 'rb') as f_in:
            f_in.seek(21)
            data = f_in.read()
            f_in.close()
            f_in = None

        with open(abs_path + '.nc', 'wb') as f_out:
            f_out.write(data)
            f_out.close()
            f_out = None

    if (abs_path[-3:] != '.nc'):
        abs_path = abs_path + '.nc'

    return abs_path



def print_file_format(abs_path):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    print(fh.file_format)
    fh.close()



def print_dimensions(abs_path):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    for dim in fh.dimensions.keys():
        print(dim)
    fh.close()



def print_variables(abs_path):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    for var in fh.variables.keys():
        print(var)
    fh.close()



def print_variable(abs_path, key):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')
    var = fh.variables[key]
    print(var)
    fh.close()



def get_var_shape(abs_path, key):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')
    shape = fh.variables[key].shape
    fh.close()

    return shape



def get_sat_metadata(abs_path):
    data_dict = {}
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')
    var = fh.variables['goes_imager_projection']

    data_dict['long_name'] = var.long_name
    data_dict['lat_0'] = var.latitude_of_projection_origin
    data_dict['lon_0'] = var.longitude_of_projection_origin
    data_dict['semi_major_axis'] = var.semi_major_axis
    data_dict['semi_minor_axis'] = var.semi_minor_axis
    data_dict['height'] = var.perspective_point_height
    data_dict['inv_flattening'] = var.inverse_flattening
    data_dict['sweep_ang_axis'] = var.sweep_angle_axis

    fh.close()
    fh = None

    return data_dict



def read_file(abs_path, window=False, meta=False):
    data_dict = {}
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    if (meta):
        data_dict['long_name'] = fh.variables['goes_imager_projection'].long_name
        data_dict['lat_0'] = fh.variables['goes_imager_projection'].latitude_of_projection_origin
        data_dict['lon_0'] = fh.variables['goes_imager_projection'].longitude_of_projection_origin
        data_dict['semi_major_axis'] = fh.variables['goes_imager_projection'].semi_major_axis
        data_dict['semi_minor_axis'] = fh.variables['goes_imager_projection'].semi_minor_axis
        data_dict['height'] = fh.variables['goes_imager_projection'].perspective_point_height
        data_dict['inv_flattening'] = fh.variables['goes_imager_projection'].inverse_flattening
        data_dict['sweep_ang_axis'] = fh.variables['goes_imager_projection'].sweep_angle_axis

    data_dict['x'] = fh.variables['x'][:]
    data_dict['y'] = fh.variables['y'][:]

    if (window is not False):
        data_dict['data'] = fh.variables['Flash_extent_density_window'][:]
    else:
        data_dict['data'] = fh.variables['Flash_extent_density'][:]

    fh.close()
    fh = None

    glm_obj = LocalGLMFile(f_path)
    glm_obj.set_data(data_dict)

    return glm_obj



def georeference(x, y, sat_lon, sat_height, sat_sweep):
    """
    Calculates the longitude and latitude coordinates of each data point

    Parameters
    ------------
    x : list
    y : list
    data : 2d array
    sat_lon : something
    sat_height : a number of some sort
    sat_sweep : i honestly dont know


    Returns
    ------------
    (lons, lats) : tuple of lists of floats
        Tuple containing a list of data longitude coordinates and a list of
        data latitude coordinates
    """

    Xs = x * sat_height
    Ys = y * sat_height

    p = pyproj.Proj(proj='geos', h=sat_height, lon_0=sat_lon, sweep=sat_sweep)

    lons, lats = np.meshgrid(Xs, Ys)
    lons, lats = p(lons, lats, inverse=True)

    return (lons, lats)



def idx_of_nearest(coords, val):
    X = np.abs(coords.flatten()-val)
    idx = np.where(X == X.min())
    idx = idx[0][0]
    return coords.flatten()[idx]



def create_bbox(lats, lons, point1, point2):
    bbox = {}

    p_lons = np.array([point1[1], point2[1]])
    p_lats = np.array([point1[0], point2[0]])

    lon1 = idx_of_nearest(lons, p_lons[0])
    lon2 = idx_of_nearest(lons, p_lons[1])

    lat1 = idx_of_nearest(lats, p_lats[0])
    lat2 = idx_of_nearest(lats, p_lats[1])

    if (lon1[1] > lon2[1]):
        bbox['max_lon'] = lon1
        bbox['min_lon'] = lon2
    else:
        bbox['max_lon'] = lon2
        bbox['min_lon'] = lon1

    if (lat1[1] > lat2[1]):
        bbox['max_lat'] = lat1
        bbox['min_lat'] = lat2
    else:
        bbox['max_lat'] = lat2
        bbox['min_lat'] = lat1

    return bbox



def plot_mercator(data_dict, extent_coords):

    globe = ccrs.Globe(semimajor_axis=data_dict['semi_major_axis'], semiminor_axis=data_dict['semi_minor_axis'],
                       flattening=None, inverse_flattening=data_dict['inv_flattening'])

    ext_lats = [extent_coords[0][0], extent_coords[1][0]]
    ext_lons = [extent_coords[0][1], extent_coords[1][1]]

    Xs, Ys = georeference(data_dict['x'], data_dict['y'], data_dict['lon_0'], data_dict['height'],
                          data_dict['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    ax.set_extent([min(ext_lons), max(ext_lons), min(ext_lats), max(ext_lats)], crs=ccrs.PlateCarree())

    cmesh = plt.pcolormesh(Xs, Ys, data_dict['data'], vmin=0, vmax=350, transform=ccrs.PlateCarree(), cmap=cm.jet)
    cbar = plt.colorbar(cmesh,fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
