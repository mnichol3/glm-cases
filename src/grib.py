"""
Author: Matt Nicholson

Functions to read & manipulate Multi-Radar/Multi-Sensor System (MRMS) GRIB files
"""
import numpy as np
import pygrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from cartopy.feature import NaturalEarthFeature
from os.path import join, isfile
import sys
from os import listdir, walk
import re
from datetime import datetime

from mrmsgrib import MRMSGrib


def print_keys(fname, keyword=None):
    """
    Prints the keys in a grib file. If the parameter 'keyword' is specified,
    it will only print keys that contain the string passed as keyword

    Parameters
    ----------
    fname : str
        Absolute path & filename of a Grib2 MRMS file to open
    keyword : str, optional
        String to look through the keys for. If specified, only keys containing
        keyword will be printed

    Returns
    -------
    None
    """

    grb = pygrib.open(fname)
    grb = grb[1]

    if (keyword is not None):
        for key in grb.keys():
            if keyword in key:
                print(key)
    else:
        for key in grb.keys():
            print(key)



def get_keys(fname, keyword=None):
    """
    Returns the keys in a grib file. If the parameter 'keyword' is specified,
    it will only return keys that contain the string passed as keyword

    Parameters
    ----------
    fname : str
        Absolute path & filename of a Grib2 MRMS file to open
    keyword : str, optional
        String to look through the keys for. If specified, only keys containing
        keyword will be printed

    Returns
    -------
    keys : list of str
    """
    grb = pygrib.open(fname)
    grb = grb[1]

    if (keyword is not None):
        keys = []

        for key in grb.keys():
            if keyword in key:
                keys.append(key)
        return keys
    else:
        return grb.keys()



def get_grb_data(abs_path, point1, point2, missing=0, debug=False):
    """
    Opens a MRMS Grib2 data file and creates a new MRMSGrib object.

    Parameters
    ----------
    abs_path : str
        The absolute path of the Grib2 file to open
    debug : bool, optional
        If True, the function prints some file metadata

    Returns
    -------
    MRMSGrib object
    """
    memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'

    grid = init_grid()
    point1, point2 = _augment_coords(point1, point2)
    min_lat, max_lat, min_lon, max_lon = get_bbox_indices(grid, point1, point2)
    grid_lons, grid_lats = subset_grid(grid, [min_lat, max_lat, min_lon, max_lon]) # (lons, lats)

    grb_file = pygrib.open(abs_path)
    grb = grb_file[1]

    data = grb.values[max_lat : min_lat, min_lon : max_lon + 1] # changed from max_lon + 1

    if (missing == 0):
        data[data < 0] = 0
    elif (missing == 'nan'):
        data[data < 0] = float(nan)
    else:
        raise ValueError('Invalid missing data argument (grib.subset_data)')

    path, fname = abs_path.rsplit('/', 1)
    data_shape = data.shape

    fp = np.memmap(join(memmap_path, fname.replace('grib2', 'txt')), dtype='float32', mode='w+', shape=data_shape)
    fp[:] = data[:]
    del fp

    major_ax = grb.earthMajorAxis
    minor_ax = grb.earthMinorAxis
    val_date = grb.validityDate
    val_time = grb.validityTime

    del grb

    grb_file.close()
    grb_file = None

    if (debug):
        print('data array shape (y, x):', data.shape)
        print('validity date:', val_date)
        print('validity time:', val_time)
        print('major axis:', major_ax)
        print('minor axis:', minor_ax)
        print('------------------------------------')

    return MRMSGrib(val_date, val_time, major_ax, minor_ax, path, fname, data_shape, grid_lons=grid_lons, grid_lats=grid_lats)



def plot_grb(grb):
    """
    Takes a MRMSGrib object and plots it on a Mercator projection

    Parameters
    ----------
    grb : MRMSGrib object

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6)) #dpi = 200

    globe = ccrs.Globe(semimajor_axis=grb.major_axis, semiminor_axis=grb.minor_axis,
                       flattening=None)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    ax.set_extent([min(grb.grid_lons), max(grb.grid_lons), min(grb.grid_lats), max(grb.grid_lats)], crs=ccrs.PlateCarree())

    cmesh = plt.pcolormesh(grb.grid_lons, grb.grid_lats, grb.data, transform=ccrs.PlateCarree(), cmap=cm.gist_ncar)

    lon_ticks = [x for x in range(-180, 181)]
    lat_ticks = [x for x in range(-90, 91)]

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='gray',
                      alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right=False
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(lat_ticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'red', 'weight': 'bold'}

    cbar = plt.colorbar(cmesh,fraction=0.046, pad=0.04)

    # Increase font size of colorbar tick labels
    plt.title('MRMS Reflectivity ' + str(grb.validity_date) + ' ' + str(grb.validity_time) + 'z')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=12)
    cbar.set_label('Reflectivity (dbz)', fontsize = 14, labelpad = 20)

    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    #fig.savefig(join(out_path, scan_date.strftime('%Y'), scan_date.strftime('%Y%m%d-%H%M')) + '.png', dpi=500)

    plt.show()



def get_files_in_dir(path):
    """
    Returns a list of all the files in the specified directory

    Parameters
    ----------
    path : str
        Path of the directory to inspect

    Returns
    -------
    files : list of str
        List of the filenames found in the directory
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]

    return files



def grid_info(fname):
    """
    Opens a MRMS Grib data file and prints various metadata

    Parameters
    ----------
    fname : str
        Absolute path of the MRMS Grib2 file to open

    Returns
    -------
    None

    """
    grb = pygrib.open(fname)
    grb = grb[1]

    print('Grid type:', grb.gridType)
    print('Grid Description Section Present:', grb.gridDescriptionSectionPresent)
    print('Grid Definition Template Number:', grb.gridDefinitionTemplateNumber)
    print('Grid Definition Description:', grb.gridDefinitionDescription)
    print('Grid Longitudes (First, Last):', grb.longitudeOfFirstGridPointInDegrees, grb.longitudeOfLastGridPointInDegrees)
    print('Grid Latitudes (First, Last):', grb.latitudeOfFirstGridPointInDegrees, grb.latitudeOfLastGridPointInDegrees)



def init_grid(debug=None):
    """
    Initialized a CONUS MRMS grid consisting of two lists, the first  holding
    longitude coordinates and the second holding latitude coordinates

    Parameters
    ----------
    debug : bool, optional
        If True, grid metadata is printed

    Returns
    -------
    Tuple of lists of float
        Tuple containing the list of longitude coordinates and the list of latitude
        coordinates. Format: (lons, lats)

    """
    inc = 0.01
    lons = np.arange(-129.995, -60.005, inc) # -129.995 to -60.005
    lats = np.arange(54.995, 19.995, inc * -1) # 54.995 to 20.005
    #grid = np.meshgrid(lons, lats)

    lons = trunc(lons, 3)
    lats = trunc(lats, 3)

    if (debug):
        print("Lons length:", len(lons))
        print(lons)
        print("Lats length:", len(lats))
        print(lats)
        print('------------------------------------')


    return (lons, lats)



def get_bbox_indices(grid, point1, point2, debug=False):
    """
    Searches through the grid to find the indices of the gridpoints corresponding
    to the bounding box formed by point1 and point2

    Parameters
    ----------
    grid : tuple of lists of float
        Format : (lons, lats)
    point1 : tuple of floats
        Coordinates of the first point to find.
        Format: (lon, lat)
    point2: tuple of floats
        Coordinates of the second point to find.
        Format: (lon, lat)
    debug : bool, optional
        If True, the found indices corresponding to the points are printed

    Returns
    -------
    indices : dict
        Dictionary containing the min lat, max lat, min lon, & max lon of the
        bounding box
        Keys: min_lon, max_lon, min_lat, max_lat

    """

    grid_lons = grid[0]
    grid_lats = grid[1]

    lats = np.array([point1[0], point2[0]])
    lons = np.array([point1[1], point2[1]])

    min_lon = np.where(grid_lons == np.amin(lons))[0][0]
    max_lon = np.where(grid_lons == np.amax(lons))[0][0]

    min_lat = np.where(grid_lats == np.amin(lats))[0][0]
    max_lat = np.where(grid_lats == np.amax(lats))[0][0]

    #indices = {'min_lon': min_lon[0][0], 'max_lon': max_lon[0][0], 'min_lat': min_lat[0][0], 'max_lat': max_lat[0][0]}

    if (debug):
        print('min lon idx:', indices['min_lon'])
        print('max lon idx:', indices['max_lon'])
        print('min lat idx:', indices['min_lat'])
        print('max lat idx:', indices['max_lat'])
        print('------------------------------------')

    #return indices
    return (min_lat, max_lat, min_lon, max_lon)



def trunc(vals, decs=0):
    """
    Truncates a list of floats to 3 decimal places

    Parameters
    ----------
    vals : list of float
        List of values to be truncated
    decs : int
        The decimal place to truncate to. Default is 0

    Returns
    -------
    List of floats

    """
    return np.trunc(vals*10**decs)/(10**decs)



def subset_grid(grid, bbox, debug=False):
    """
    Creates a subset of the CONUS MRMS grid defined by the bounding box

    Parameters
    ----------
    grid : tuple of lists of float
        Format : (lons, lats)
    bbox : dict
        Dictionary containing the min lat, max lat, min lon, & max lon of the
        bounding box
        Keys: min_lon, max_lon, min_lat, max_lat
    debug : bool, optional
        If True, the length of the new lat & lon coordinate lists are printed

    Returns
    -------
    Tuple of lists
        Tuple containing the new grid latitude & longitude values
        Format: (lons, lats)

    """
    x_min = bbox[2]
    x_max = bbox[3]
    y_min = bbox[0]
    y_max = bbox[1]

    lons = grid[0][x_min : x_max + 1]
    lats = grid[1][y_max : y_min]

    if (debug):
        print('lons (x) length:', len(lons))
        print('lats (y) length:', len(lats))
        print('------------------------------------')

    return (lons, lats)



def subset_data(bbox, data, missing=0, debug=False):
    """
    Extracts a subset of data from the MRMS CONUS data file

    Parameters
    ----------
    bbox : dict
        Dictionary containing the min lat, max lat, min lon, & max lon of the
        bounding box
        Keys: min_lon, max_lon, min_lat, max_lat
    data : numpy 2d array
        2d array containing CONUS MRMS data for a single scan angle
    missing : int or str
        Indicates what to replace missing, or emply, grid cells with. They are
        natively assigned the value -999. Only excepts 0 or 'nan', otherwise a
        ValueError is raised. 0 is ideal for cross-sectional analysis, while nan
        is ideal for plan-view plotting
    debug : bool, optional
        If True, metadata pertaining to the new 2d data array is printed

    Returns
    -------
    subset : numpy 2d array
        2d array containing data from the subset defined by the bounding box

    """

    x_min = bbox['min_lon']
    x_max = bbox['max_lon']
    y_min = bbox['min_lat']
    y_max = bbox['max_lat']

    subset = data[y_max : y_min, x_min : x_max + 1]

    if (missing == 0):
        subset[subset < 0] = 0
    elif (missing == 'nan'):
        subset[subset < 0] = float(nan)
    else:
        raise ValueError('Invalid missing data argument (grib.subset_data)')

    if (debug):
        print('min x idx:', x_min)
        print('max x idx:', x_max)
        print('min y idx:', y_min)
        print('max y idx:', y_max)

        print('subset shape (y, x):', subset.shape)
        print('------------------------------------')

    return subset



def fetch_scans(base_path, time, angles=None):
    """
    Looks through the subdirectories in base_path and returns all the files with
    the same validity time as the parameter 'time'

    Parameters
    ----------
    base_path : str
        Path to the directory that holds the subdirectories we wish to look through
    time : str
        The validity time of the scans
        Format: HHMM or HH:MM
    angles : list of str, optional
        Scan angles to limit search results to

    Returns
    -------
    scans : list of str
        Sorted list of filenames found in the subdirectories that have the desired
        validity time

    """
    scans = []
    time_re = re.compile(r'-(\d{4})')
    scan_re = re.compile(r'_(\d{2}.\d{2})_')

    if (isinstance(time, int)):
        time = str(time)
    else:
        if (':' in time):
            hours, mins = time.split(':')
            time = hours + mins

    for subdir, dirs, files in walk(base_path):
        curr_fname = None
        curr_tdelta = 100000
        for file in files:
            time_match = time_re.search(file)
            if (time_match is not None):
                found_time = time_match.group(1)
                if (angles):
                    angle_match = scan_re.search(file)
                    if (angle_match is not None and angle_match.group(1) in angles and found_time == time):
                        scans.append(file)
                else:
                    if (found_time == time):
                        scans.append(file)
                    else:
                        FMT = '%H%M'
                        tdelta = (datetime.strptime(found_time, FMT) - datetime.strptime(time, FMT)).total_seconds()
                        tdelta = abs(tdelta)
                        if (tdelta < abs(curr_tdelta)):
                            curr_tdelta = tdelta
                            curr_fname = file
        if (curr_fname):
            scans.append(curr_fname)

    return sorted(scans)



def parse_fname(base_path, fname):
    """
    Creates an absolute path to an MRMS Grib data file

    Parameters
    ----------
    base_path : str
        The base path that hold the various scan angle subdirectories
    fname : str
        Name of the file in the subdirectory

    Returns
    -------
    abs_path : str
        Absolute path of the file

    """
    base_path += '/MergedReflectivityQC_'

    scan_re = re.compile(r'_(\d{2}.\d{2})_')
    match = scan_re.search(fname)

    if (match is not None):
        base_path += match.group(1)
    else:
        print('Error parsing filename')
        sys.exit(0)

    abs_path = join(base_path, fname)

    return abs_path



def get_grib_objs(scans, base_path, point1, point2):
    """
    Creates and returns a list of new MRMSGrib objects

    Parameters
    ----------
    scans : list of str
        MRMS Grib file names
    base_path : str
        Path to the directory that holds the subdirectories of the various scan
        angles

    Returns
    -------
    grb_files : list of MRMSGrib objects

    """
    grb_files = []

    if (not isinstance(scans, (list,))):
        scans = [scans]

    for file in scans:
        print('Parsing ', file)

        f_path = parse_fname(base_path, file)

        grid = init_grid()

        grb_file = get_grb_data(f_path, point1, point2)

        grb_files.append(grb_file)

    return grb_files



def _augment_coords(point1, point2):
    point1 = list(point1)
    point2 = list(point2)

    if (point1[0] < point2[0]):
        point1[0] -= 1
        point2[0] += 1
    else:
        point2[0] -= 1
        point1[0] += 1

    if (point1[1] < point2[1]):
        point1[1] -= 1
        point2[1] += 1
    else:
        point2[1] -= 1
        point1[1] += 1

    return (point1, point2)
