import numpy as np
from os.path import join, isdir, isfile
from os import mkdir
import re
from pyproj import Geod
from math import sin, cos, sqrt, atan2, radians

from grib import fetch_scans, get_grib_objs

def to_file(out_path, f_name, data):
    """
    Writes a numpy 2d array to a text file

    Parameters
    ----------
    out_path : str
        Path of the directory in which to save the text file
    f_name : str
        Desired name of the text file
    data : numpy 2d array
        Data to write to the text file

    Returns
    -------
    abs_path : str
        Absolute path of the text file
    """

    if (not isdir(out_path)):
        mkdir(out_path)

    abs_path = join(out_path, f_name)

    print("\nWriting", abs_path, "\n")

    np.savetxt(abs_path, data, delimiter=',', newline='\n', fmt='%2.3f')

    return abs_path



def load_data(abs_path):
    """
    Reads a numpy 2d array from a text file and returns it

    Parameters
    ----------
    abs_path : str
        Absolute path of the text file, including the filename

    Returns
    -------
    data : numpy 2d array of float
        2d array read from the text file
    """
    if (not isfile(abs_path)):
        raise OSError('File not found (plot_cross.load_data)')
    else:
        print('Loading MRMS cross_section data from', abs_path, '\n')
        data = np.loadtxt(abs_path, dtype=float, delimiter=',')

        return data



def load_coordinates(abs_path):
    if (not isfile(abs_path)):
        raise OSError('File not found (plot_cross.load_coordinates)')
    else:
        print('Loading MRMS cross_section coordinate data from', abs_path, '\n')
        data = np.loadtxt(abs_path, dtype=float, delimiter=',')

        return data



def parse_coord_fnames(abs_path):
    date_re = re.compile(r'(\d{8})')
    time_re = re.compile(r'(\d{4})z')
    f_base = 'mrms-cross-'

    date_match = date_re.search(abs_path)
    if (date_match is not None):
        val_date = date_match.group(1)

        time_match = time_re.search(abs_path)
        if (time_match is not None):
            val_time = time_match.group(1)

            f_lon = f_base + val_date + '-' + val_time + 'z' + '-lons.txt'

            f_lon = join(BASE_PATH_XSECT_COORDS, f_lon)

            f_lat = f_base + val_date + '-' + val_time + 'z' + '-lats.txt'

            f_lat = join(BASE_PATH_XSECT_COORDS, f_lat)

            return (f_lon, f_lat)

    raise OSError('Unable to parse coordinate file(s)')



def process_slice(base_path, slice_time, point1, point2, write=False):
    cross_sections = np.array([])

    scans = fetch_scans(base_path, slice_time) # z = 33

    grbs = get_grib_objs(scans, base_path)

    valid_date = grbs[0].validity_date
    valid_time = grbs[0].validity_time

    fname = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z.txt'

    cross_sections = np.asarray(get_cross_neighbor(grbs[0], point1, point2))

    for grb in grbs[1:]:
        cross_sections = np.vstack((cross_sections, get_cross_neighbor(grb, point1, point2)))

    if (write):
        f_out = to_file(BASE_PATH_XSECT, fname, cross_sections)
        return f_out
    else:
        return cross_sections



def process_slice_inset(base_path, slice_time, point1, point2):
    """
    ex:
        dict = process_slice2(base_path, slice_time, point1, point2)
        plot_cross_section_inset(inset_data=dict['f_inset_data'], inset_lons=dict['f_inset_lons'],
            inset_lats=dict['f_inset_lats'], abs_path=fname, points=(point1, point2))
    """
    cross_sections = np.array([])

    scans = fetch_scans(base_path, '2124') # z = 33

    grbs = get_grib_objs(scans, base_path)

    valid_date = grbs[0].validity_date
    valid_time = grbs[0].validity_time

    fname = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z.txt'

    cross_sections = np.asarray(get_cross_neighbor(grbs[0], point1, point2))

    for grb in grbs[1:]:
        cross_sections = np.vstack((cross_sections, get_cross_neighbor(grb, point1, point2)))

    ang2 = 'mrms-ang2-' + str(valid_date) + '-' + str(valid_time) + 'z.txt'
    f_ang2_lons = 'mrms-ang2-' + str(valid_date) + '-' + str(valid_time) + 'z-lons.txt'
    f_ang2_lats = 'mrms-ang2-' + str(valid_date) + '-' + str(valid_time) + 'z-lats.txt'

    f_out = to_file(BASE_PATH_XSECT, fname, cross_sections)
    f_lons = to_file(BASE_PATH_XSECT_COORDS, f_ang2_lons, grbs[6].grid_lons)
    f_lats = to_file(BASE_PATH_XSECT_COORDS, f_ang2_lats, grbs[6].grid_lats)
    f_inset = to_file(BASE_PATH_XSECT, ang2, grbs[6].data)

    return {'x_sect': f_out, 'f_inset_lons': f_lons, 'f_inset_lats': f_lats, 'f_inset_data': f_inset}



def calc_geod_pts(point1, point2, num_pts):
    """
    Calculates a number of points, num_pts, along a line defined by point1 & point2

    Parameters
    ----------
    point1 : tuple of floats
        First geographic coordinate pair
        Format: (lat, lon)
    point2 : tuple of floats
        Second geographic coordinate pair
        Format: (lat, lon)
    num_pts : int
        Number of coordinate pairs to calculate

    Returns
    -------
    Yields a tuple of floats
    Format: (lon, lat)
    """
    geod = Geod("+ellps=WGS84")
    points = geod.npts(lon1=point1[1], lat1=point1[0], lon2=point2[1],
                   lat2=point2[0], npts=num_pts)

    for pt in points:
        yield pt



def calc_dist(point1, point2, units='km'):
    """
    Calculates the distance between two geographic coordinates in either km,
    the default, or in meters

    Parameters
    ----------
    point1 : tuple of floats
        First point
        Format: (lat, lon)
    point2 : tuple of floats
        Second point
        Format: (lat, lon)
    units : str, optional
        If units = m, the distance will be returned in meters instead of kilometers
    """
    R = 6373.0  # Approx. radius of Earth, in km

    lat1 = radians(point1[0])
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    if (units == 'm'):
        distance = distance * 1000

    return distance



def filter_by_dist(lma_df, dist, start_point, end_point, num_pts):
    """
    Filters the WTLMA dataframe to only include events that are within a certain
    distance of the line that defines the MRMS cross-section

    Parameters
    ----------
    lma_dt : Pandas DataFrame
        DataFrame containing the WTLMA data.
        Columns: 'time', 'lat', 'lon', 'alt', 'r chi2', 'P', 'mask'
    dist: int
        Distance threshold
    start_point : tuple of floats
        Coordinates of the point defining the beginning of the cross-section.
        Format: (lat, lon)
    end_point : tuple of floats
        Coordinates of the point defining the end of the cross-section.
        Format: (lat, lon)
    num_pts : int
        Number of geographic coordinate pairs to calculate between start_point &
        end_point

    Returns
    -------
    subs_df : Pandas DataFrame
        DataFrame containing the filtered WTLMA events
    """
    s_lat = start_point[0]
    s_lon = start_point[1]
    e_lat = end_point[0]
    e_lon = end_point[1]

    idxs = []

    for pt1 in calc_geod_pts(start_point, end_point, num_pts=num_pts):
        for idx, pt2 in enumerate(list(zip(lma_df['lat'].tolist(), lma_dt['lon'].tolist()))):
            # reverse the order of pt1 since the function returns the coordinates
            # as (lon, lat) and calc_dist wants (lat, lon)
            if (calc_dist((pt1[1], pt1[0]), pt2, units='m') <= 300):
                idxs.append(idx)

    subs_df = lma_df.iloc[idxs]

    return subs_df
