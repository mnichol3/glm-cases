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

from grib import trunc

def trim_header(abs_path):
    if (not isfile(abs_path)):
        raise OSError('File does not exist:', abs_path)

    if (not isfile(abs_path + '.nc') and abs_path[-3:] != '.nc'):

        with open(abs_path, 'rb') as f_in:
            f_in.seek(21)
            data = f_in.read()
            f_in.close()
            f_in = None

        with open(out_path, 'wb') as f_out:
            f_out.write(data)
            f_out.close()
            f_out = None

    if (abs_path[-3:] != '.nc'):
        abs_path = abs_path + '.nc'

    return abs_path



def get_axis_range(coords):
    maxs = []
    mins = []
    for x in coords:
        maxs.append(max(x))
        mins.append(min(x))

    while (1e+30 in maxs):
        maxs.remove(1e+30)

    while (1e+30 in mins):
        mins.remove(1e+30)

    abs_min = min(mins)
    abs_max = max(maxs)

    return (abs_min, abs_max)



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



def read_file(abs_path, meta=False):
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
    data_dict['data'] = fh.variables['Flash_extent_density'][:]

    fh.close()
    fh = None

    return data_dict



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

    # Multiplying by sat height might not be necessary here
    Xs = x * sat_height # (1000,)
    Ys = y * sat_height # (1000,)

    p = pyproj.Proj(proj='geos', h=sat_height, lon_0=sat_lon, sweep=sat_sweep)

    lons, lats = np.meshgrid(Xs, Ys)
    lons, lats = p(lons, lats, inverse=True)

    return (lons, lats)



def idx_of_nearest(coords, val):
    X = np.abs(coords-val)
    idx = np.where(X == X.min())
    return (idx, coords[idx][0])



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

    cmesh = plt.pcolormesh(Xs, Ys, data_dict['data'], transform=ccrs.PlateCarree(), cmap=cm.jet)
    cbar = plt.colorbar(cmesh,fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def main():
    fname = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523/IXTR99_KNES_232122_14654.2019052322'
    #fname = '/media/mnichol3/pmeyers1/MattNicholson/abi/ABI-L2-CMIPF_2019_143_21_OR_ABI-L2-CMIPF-M6C06_G16_s20191432130383_e20191432140097_c20191432140168.nc'

    data = read_file(fname, meta=True)

    #lons, lats = georeference(meta['x'], meta['y'], meta['lon_0'], meta['height'], meta['sweep_ang_axis'])

    point1 = (37.195, -102.185)
    point2 = (34.565, -99.865)

    """
    bbox = create_bbox(lats, lons, point1, point2)
    print(bbox)
    print(lons.shape)
    lons_sub = lons[890:1403, 560:570]
    lats_sub = lats[509:554, 263:1847]
    """


    """
    plt.figure(figsize=[15,12])
    plt.axis([-102.185, -99.865, 34.656, 37.195])
    mesh = plt.pcolormesh(lons, lats, meta['data'])
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.plot(lons.flatten(), lats.flatten(), c='r')
    plt.show()
    """
    plot_mercator(data, (point1, point2))




if (__name__ == '__main__'):
    main()
