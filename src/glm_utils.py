from os.path import isfile
from netCDF4 import Dataset
import pyproj
import numpy as np
from math import pi
import matplotlib.pyplot as plt

def trim_header(abs_path):
    if (not isfile(abs_path)):
        raise OSError('File does not exist:', abs_path)

    out_path = abs_path + '.nc'

    print(out_path)

    if (not isfile(out_path)):

        with open(abs_path, 'rb') as f_in:
            f_in.seek(21)
            data = f_in.read()
            f_in.close()
            f_in = None

        with open(out_path, 'wb') as f_out:
            f_out.write(data)
            f_out.close()
            f_out = None

    return out_path



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



def georeference(x, y, data, sat_lon, sat_height, sat_sweep):
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

    lats[np.isnan(data)] = 57
    lons[np.isnan(data)] = -152

    return (lons, lats)



fname = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523/IXTR99_KNES_232122_14654.2019052322'
print_variable(fname, 'x')

"""
meta = read_file(fname, meta=True)
lons, lats = georeference(meta['x'], meta['y'], meta['data'], meta['lon_0'], meta['height'], meta['sweep_ang_axis'])
plt.figure(figsize=[15,12])
#plt.pcolormesh(meta['x'], meta['y'], meta['data']) # Works in geos projection
plt.pcolormesh(lons, lats, meta['data'])
plt.show()
"""
