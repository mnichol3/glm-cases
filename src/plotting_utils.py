import numpy as np
from os.path import join, isdir, isfile
from os import mkdir
import re
from pyproj import Geod
from math import sin, cos, sqrt, atan2, radians
from sys import exit
from functools import partial
from shapely.geometry import Point, LineString
from shapely.ops import transform
import pyproj
import sys
import math
import datetime

from grib import fetch_scans, get_grib_objs
from mrmscomposite import MRMSComposite


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



def get_cross_cubic(grb, point1, point2, first=False):
    """
    Calculates the cross section of a single MRMSGrib object's data from point1 to point2
    using cubic interpolation

    Parameters
    ----------
    grb : MRMSGrib object
    point1 : tuple of float
        Coordinates of the first point that defined the cross section
        Format: (lat, lon)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lat, lon)
    first : bool, optional
        If True, the cross section latitude & longitude coordinates will be calculated
        and written to text files

    Returns
    -------
    zi : numpy nd array
        Array containing cross-section reflectivity
    """
    BASE_PATH = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    BASE_PATH_XSECT = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'
    BASE_PATH_XSECT_COORDS = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/coords'
    lons = grb.grid_lons
    lats = grb.grid_lats

    x, y = np.meshgrid(lons, lats)
    z = grb.data

    # [(x1, y1), (x2, y2)]
    line = [(point1[0], point1[1]), (point2[0], point2[1])]

    # cubic interpolation
    y_world, x_world = np.array(list(zip(*line)))
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y.max() - y_world ) / y.ptp()

    num = 100
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    valid_date = grb.validity_date
    valid_time = grb.validity_time

    if (first):

        fname_lons = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lons.txt'
        fname_lats = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lats.txt'

        d_lats, d_lons = calc_coords(point1, point2, num)

        to_file(BASE_PATH_XSECT + '/coords', fname_lons, d_lons)
        to_file(BASE_PATH_XSECT + '/coords', fname_lats, d_lats)

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(z, np.vstack((row, col)), order=1, mode='nearest')

    return zi



def get_cross_neighbor(grb, point1, point2, first=False):
    """
    Calculates the cross section of a single MRMSGrib object's data from point1 to point2
    using nearest-neighbor interpolation

    Parameters
    ----------
    grb : MRMSGrib object
    point1 : tuple of float
        Coordinates of the first point that defined the cross section
        Format: (lat, lon)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lat, lon)
    first : bool, optional
        If True, the cross section latitude & longitude coordinates will be calculated
        and written to text files

    Returns
    -------
    zi : numpy nd array
        Array containing cross-section reflectivity
    """
    BASE_PATH = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    BASE_PATH_XSECT = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'
    BASE_PATH_XSECT_COORDS = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/coords'
    lons = grb.grid_lons
    lats = grb.grid_lats

    x, y = np.meshgrid(lons, lats)
    #z = grb.data
    z = np.memmap(grb.get_data_path(), dtype='float32', mode='r', shape=grb.shape)

    line = [(point1[0], point1[1]), (point2[0], point2[1])]

    y_world, x_world = np.array(list(zip(*line)))

    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y.max() - y_world ) / y.ptp()

    num = 1000
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    valid_date = grb.validity_date
    valid_time = grb.validity_time

    d_lats, d_lons = calc_coords(point1, point2, num)

    zi = z[row.astype(int), col.astype(int)]

    del z

    return (zi, d_lats, d_lons)



def process_slice(base_path, slice_time, point1, point2, write=False):
    """
    Does all the heavy lifting to compute a vertical cross section slice of MRMS
    data

    Parameters
    ----------
    base_path : str
        Path to the parent MRMS data directory
    slice_time : str
        Validity time of the desired data
    point1 : tuple of floats
        First coordinate pair defining the cross section
        Format: (lat, lon)
    point2 : tuple of floats
        Second coordinate pair defining the cross section
        Format: (lat, lon)
    write: bool, optional
        If true, the cross section array will be written to a file

    Returns
    -------
    Tuple
        Contains the cross section array, lats, and lons
    """
    BASE_PATH_XSECT = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'
    BASE_PATH_XSECT_COORDS = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/coords'

    cross_sections = np.array([])

    scans = fetch_scans(base_path, slice_time) # z = 33

    grbs = get_grib_objs(scans, base_path, point1, point2)

    valid_date = grbs[0].validity_date
    valid_time = grbs[0].validity_time

    fname = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z.txt'

    cross_sections, lats, lons = np.asarray(get_cross_neighbor(grbs[0], point1, point2))

    for grb in grbs[1:]:
        x_sect, _, _ = get_cross_neighbor(grb, point1, point2)
        cross_sections = np.vstack((cross_sections, x_sect))

    if (write):
        f_out = to_file(BASE_PATH_XSECT, fname, cross_sections)
        return f_out
    else:
        return (cross_sections, lats, lons)



def process_slice_inset(base_path, slice_time, point1, point2):
    """
    Does all the heavy lifting to compute a vertical cross section slice of MRMS
    data with geographical plot inset

    Parameters
    ----------
    base_path : str
        Path to the parent MRMS data directory
    slice_time : str
        Validity time of the desired data
    point1 : tuple of floats
        First coordinate pair defining the cross section
        Format: (lat, lon)
    point2 : tuple of floats
        Second coordinate pair defining the cross section
        Format: (lat, lon)

    Returns
    -------
    Dictionary
        Keys: x_sect, f_inset_lons, f_inset_lats, f_inset_data

    ex:
        dict = process_slice2(base_path, slice_time, point1, point2)
        plot_cross_section_inset(inset_data=dict['f_inset_data'], inset_lons=dict['f_inset_lons'],
            inset_lats=dict['f_inset_lats'], abs_path=fname, points=(point1, point2))
    """
    BASE_PATH_XSECT = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'
    BASE_PATH_XSECT_COORDS = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/coords'

    cross_sections = np.array([])

    scans = fetch_scans(base_path, '2124') # z = 33

    grbs = get_grib_objs(scans, base_path, point1, point2)

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
    coords : list of tuples
        List of coordinates of filtered WTLMA events (?)
        Format: (lat, lon)
    """
    if (not isinstance(dist, int)):
        raise TypeError('dist must be of type int')

    s_lat = start_point[0]
    s_lon = start_point[1]
    e_lat = end_point[0]
    e_lon = end_point[1]

    idxs = []
    coords = []
    alts = lma_df['alt'].tolist()
    xsect_az = int(calc_bearing(start_point, end_point))

    for pt1 in calc_geod_pts(start_point, end_point, num_pts=num_pts):
        for idx, pt2 in enumerate(list(zip(lma_df['lat'].tolist(), lma_df['lon'].tolist()))):
            # reverse the order of pt1 since the function returns the coordinates
            # as (lon, lat) and calc_dist wants (lat, lon)
            curr_az = int(calc_bearing((pt1[1], pt1[0]), pt2))
            if ((calc_dist((pt1[1], pt1[0]), pt2, units='m') <= dist) and (idx not in idxs) and (alts[idx] < 19000)):
                idxs.append(idx)
                coords.append([pt1[1], pt1[0]])

    # Remove repeat indexes from list
    # MUCH faster to use a set than another conditional inside the nested loops
    #idxs = list(set(idxs))
    subs_df = lma_df.iloc[idxs]

    return subs_df, coords



def calc_bearing(point1, point2):
    """
    Calculates the bearing between two points

    https://gist.github.com/jeromer/2005586

    Parameters
    ----------
    point1 : tuple of floats
        Format: (lat, lon)
    point2 : tuple of floats
        Format: (lat, lon)

    Returns
    -------
    bearing : float
    """

    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])

    diffLong = math.radians(point2[1] - point1[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    return bearing



def bearing_diff(bearing1, bearing2):
    diff = abs(bearing1 - bearing2)
    if (diff > 360):
        diff = abs(360 - diff)
    if (diff > 180):
        diff = abs(360 - diff)
    return diff



def get_composite_ref(base_path, slice_time, point1, point2, memmap_path):
    """
    Creates a composite reflectivity product from the 33 MRMS scan angles

    Parameters
    ----------
    base_path : str
        Path of the parent MRMS data directory
    slice_time : str
        Validity time of the desired data
    point1 : tuple of floats
        First coordinate pair defining the cross section
        Format: (lat, lon)
    point2 : tuple of floats
        Second coordinate pair defining the cross section
        Format: (lat, lon)
    memmap_path : str
        Path to the directory being used to store memory-mapped array files

    Returns
    -------
    comp_obj : MRMSComposite object
    """
    #memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'
    scans = fetch_scans(base_path, slice_time)

    grbs = get_grib_objs(scans, base_path, point1, point2)

    valid_date = grbs[0].validity_date
    valid_time = grbs[0].validity_time
    major_axis = grbs[0].major_axis
    minor_axis = grbs[0].minor_axis
    data_shape = grbs[0].shape

    fname = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z.txt'

    scan_0 = np.memmap(grbs[0].get_data_path(), dtype='float32', mode='r', shape=grbs[0].shape)
    composite = np.empty_like(scan_0)
    composite[:] = scan_0

    del scan_0

    for grb in grbs[1:]:
        curr_ref = np.memmap(grb.get_data_path(), dtype='float32', mode='r', shape=grb.shape)

        for idx, val in enumerate(composite):
            for sub_idx, sub_val in enumerate(val):
                if (curr_ref[idx][sub_idx] > composite[idx][sub_idx]):
                    composite[idx][sub_idx] = curr_ref[idx][sub_idx]
        del curr_ref

    fname = '{}-{}-{}'.format('comp_ref', valid_date, valid_time)
    outpath = join(memmap_path, fname)

    # write the composite data to memmap arr
    fp = np.memmap(outpath, dtype='float32', mode='w+', shape=data_shape)
    fp[:] = composite[:]
    del fp

    comp_obj = MRMSComposite(valid_date, valid_time, major_axis, minor_axis,
                             outpath, fname, data_shape, grid_lons=grbs[0].grid_lons, grid_lats=grbs[0].grid_lats)

    return comp_obj



def calc_coords(point1, point2, num):
    """
    Calculates the coordinates for a number, num, of points along the line
    defined by point1 and point2

    Parameters
    ----------
    point1 : tuple of floats
        Format: (lat, lon)
    point2 : tuple of floats
        Format: (lat, lon)

    Returns
    -------
    Tuple of lists
        Tuple containing the lists of latitude and longitude coordinates
        Format: (lats, lons)
    """
    xs = [point1[1], point2[1]]
    ys = [point1[0], point2[0]]

    lons = np.linspace(min(xs), max(xs), num)
    lats = np.linspace(min(ys), max(ys), num)

    return (lats, lons)



def geodesic_point_buffer(lat, lon, km):
    """
    Creates a circle on on the earth's surface, centered at (lat, lon) with
    radius of km. Used to form the range rings needed for plotting

    Parameters
    ------------
    lat : float
        Latitude coordinate of the circle's center

    lon : float
        Longitude coordinate of the circle's center

    km : int
        Radius of the circle, in km


    Returns
    ------------
    A list of floats that prepresent the coordinates of the circle's edges
    """

    proj_wgs84 = pyproj.Proj(init='epsg:4326')

    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres

    return transform(project, buf).exterior



def get_wwa_polys(abs_path, date, time, wwa_type=['SV', 'TO']):
    """
    Gets NWS WWA polygons for a specified date & time

    Parameters
    ----------
    abs_path : str
        Absolute path of the WWA shapefile
    date : str
        Format: MMDDYYYY
    time : str
        Format: HHMM
    wwa_type : list of str, optional
        Types of warnings to get. Default: SV (Severe Thunderstorm) & TO (Tornado)

    Returns
    -------
    polys : dict; key : str, value : polygon

    Notes
    -----
    Native shapefile datetime format: 201905232120
    """
    polys = {}
    target_dt = _format_wwa_time(date, time)
    wwa_reader = shpreader.Reader(abs_path)

    if ('SV' in wwa_type):
        filtered_wwa_sv = [rec.geometry for rec in wwa_reader.records() if (rec.attributes['GTYPE'] == 'P')
                        and (_valid_wwa_time(rec.attributes['ISSUED'], rec.attributes['EXPIRED'], target_dt))
                        and (rec.attributes['PHENOM'] == 'SV')]
        polys['SV'] = filtered_wwa_sv
    if ('TO' in wwa_type):
        filtered_wwa_to = [rec.geometry for rec in wwa_reader.records() if (rec.attributes['GTYPE'] == 'P')
                        and (_valid_wwa_time(rec.attributes['ISSUED'], rec.attributes['EXPIRED'], target_dt))
                        and (rec.attributes['PHENOM'] == 'TO')]
        polys['TO'] = filtered_wwa_sv
    return polys



def _valid_wwa_time(issued, expired, target):
    target = int(target)
    expired = int(expired)
    issued = int(issued)
    return (target >= issued and target <= expired)



def _format_wwa_time(date, time):
    """
    Formats a datetime string to filter WWA polygons

    Parameters
    ----------
    date : str
        Format: MMDDYYYY
    time : str
        Format: HHMM

    Returns
    -------
    str
        WWA polygon datetime
        Format: YYYYMMDDHHMM
    """
    dt = datetime.datetime.strptime(date + time,'%m%d%Y%H%M')
    return datetime.datetime.strftime(dt, '%Y%m%d%H%M')
