from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from cartopy.feature import NaturalEarthFeature

import goesawsinterface
from glm_utils import georeference
from proj_utils import geod_to_scan



def get_abi_files(base_path, satellite, product, t_start, t_end, sector, channel, prompt=False):
    """
    Fetches GOES ABI file(s) for the given satellite, product, sector, & channel,
    for the time period spanning t_start to t_end

    Parameters
    ----------
    base_path : str
        Path of the directory to download the files to
    satellite : str
        'goes16' or 'goes17'
    product : str
        Just use 'ABI-L2-CMIPM'
    t_start : str
        start time
        Format: MM-DD-YYYY-HH:MM
    t_end : str
        End time
        Format: MM-DD-YYYY-HH:MM
    sector : str
        M1 = mesoscale 1, M2 = mesoscale 2, C = CONUS
    channel : str
        Imagery channel
    prompt : bool, optional
        If true, will prompt the user before downloading the ABI files

    Returns
    -------
    abi_paths : list of str
        List containing the absolute paths of the downloaded ABI files
    """
    abi_paths = []
    conn = goesawsinterface.GoesAWSInterface()
    aws_abi_files = conn.get_avail_images_in_range(satellite, product, t_start, t_end, sector, channel)

    for f in aws_abi_files:
        print(f)

    if (prompt):
        dl = input('Download the above ABI file(s)? (Y/n) ')
        if (dl != 'Y'):
            print('Proceed response not entered, exiting...')
            sys.exit(0)
    else:
        dl_results = conn.download(satellite, aws_abi_files, base_path, keep_aws_folders=False, threads=6)

        for abi_file in dl_results._successfiles:
            abi_paths.append(abi_file.filepath)

    return abi_paths



def read_file(abi_file, extent=None):
    """
    Opens & reads a GOES-16 ABI data file, returning a dictionary of data

    Parameters:
    ------------
    abi_file : str
        Absolute path of the GOES-16 ABI file to be opened & processed
    extent : list of float, optional
        List of floats used to subset the ABI data
        Format: [min_lat, max_lat, min_lon, max_lon]


    Returns:
    ------------
    data_dict : dictionary of str
        Dictionar of ABI image data & metadata from the netCDF file
    """

    data_dict = {}

    fh = Dataset(abi_file, mode='r')

    data_dict['band_id'] = fh.variables['band_id'][0]

    data_dict['band_wavelength'] = "%.2f" % fh.variables['band_wavelength'][0]
    data_dict['semimajor_ax'] = fh.variables['goes_imager_projection'].semi_major_axis
    data_dict['semiminor_ax'] = fh.variables['goes_imager_projection'].semi_minor_axis
    data_dict['inverse_flattening'] = fh.variables['goes_imager_projection'].inverse_flattening
    data_dict['latitude_of_projection_origin'] = fh.variables['goes_imager_projection'].latitude_of_projection_origin
    data_dict['longitude_of_projection_origin'] = fh.variables['goes_imager_projection'].longitude_of_projection_origin
    data_dict['data_units'] = fh.variables['CMI'].units

    # Seconds since 2000-01-01 12:00:00
    add_seconds = fh.variables['t'][0]

    # Datetime of scan
    scan_date = datetime(2000, 1, 1, 12) + timedelta(seconds=float(add_seconds))

    # Satellite height
    sat_height = fh.variables['goes_imager_projection'].perspective_point_height

    # Satellite longitude & latitude
    sat_lon = fh.variables['goes_imager_projection'].longitude_of_projection_origin
    sat_lat = fh.variables['goes_imager_projection'].latitude_of_projection_origin

    # Satellite lat/lon extend
    lat_lon_extent = {}
    lat_lon_extent['n'] = fh.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude
    lat_lon_extent['s'] = fh.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
    lat_lon_extent['e'] = fh.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
    lat_lon_extent['w'] = fh.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude

    # Geospatial lat/lon center
    data_dict['lat_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lat_center
    data_dict['lon_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lon_center

    # Satellite sweep
    sat_sweep = fh.variables['goes_imager_projection'].sweep_angle_axis

    X = fh.variables['x'][:]
    Y = fh.variables['y'][:]

    data_dict['scan_date'] = scan_date
    data_dict['sat_height'] = sat_height
    data_dict['sat_lon'] = sat_lon
    data_dict['sat_lat'] = sat_lat
    data_dict['lat_lon_extent'] = lat_lon_extent
    data_dict['sat_sweep'] = sat_sweep

    if (extent is not None):
        Xs, Ys = georeference(X, Y, sat_lon, sat_height, sat_sweep)
        min_y, max_y, min_x, max_x = subset_grid(extent, Xs, Ys)

        data = fh.variables['CMI'][min_y : max_y, min_x : max_x].data # might not work idk
        Xs = Xs[min_x : max_x]
        Ys = Ys[min_y : max_y]
    else:
        print('WARNING: Not subsetting ABI data!\n')

    fh.close()
    fh = None

    data_dict['scan_date'] = scan_date
    data_dict['sat_height'] = sat_height
    data_dict['sat_lon'] = sat_lon
    data_dict['sat_lat'] = sat_lat
    data_dict['lat_lon_extent'] = lat_lon_extent
    data_dict['sat_sweep'] = sat_sweep
    data_dict['x'] = Xs
    data_dict['y'] = Ys
    data_dict['data'] = data

    return data_dict



def plot_geos(data_dict):
    """
    Plot the GOES-16 ABI file on a geostationary-projection map

    Parameters
    ------------
    data_dict : dictionary
        Dictionary of data & metadata from GOES-16 ABI file


    Returns
    ------------
    A plot of the ABI data on a geostationary-projection map

    The projection x and y coordinates equals the scanning angle (in radians)
    multiplied by the satellite height
    http://proj4.org/projections/geos.html <-- 404'd
    https://proj4.org/operations/projections/geos.html

    """

    sat_height = data_dict['sat_height']
    sat_lon = data_dict['sat_lon']
    sat_sweep = data_dict['sat_sweep']
    scan_date = data_dict['scan_date']
    data = data_dict['data']

    Xs = data_dict['x'] * sat_height
    Ys = data_dict['y'] * sat_height

    X, Y = np.meshgrid(Xs,Ys)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(central_longitude=sat_lon,
                                satellite_height=sat_height,false_easting=0,false_northing=0,
                                globe=None, sweep_axis=sat_sweep))

    ax.set_xlim(min(Xs), max(Xs))
    ax.set_ylim(min(Ys), max(Ys))


    ax.coastlines(resolution='10m', color='gray')
    plt.title('GOES-16 Imagery', fontweight='semibold', fontsize=15)
    plt.title('%s' % scan_date.strftime('%H:%M UTC %d %B %Y'), loc='right')
    plt.pcolormesh(X, Y, data, cmap=cm.Greys_r)

    cent_lat = 29.93
    cent_lon = -71.35

    plt.scatter(cent_lon,cent_lat, marker="+", color="r", transform=ccrs.PlateCarree(),
                s = 200)

    plt.show()



def plot_mercator(data_dict, out_path):
    """
    Plot the GOES-16 data on a lambert-conformal projection map. Includes ABI
    imagery, GLM flash data, 100km, 200km, & 300km range rings, and red "+" at
    the center point

    Parameters
    ------------
    data_dict : dictionary
        Dictionary of data & metadata from GOES-16 ABI file


    Returns
    ------------
    A plot of the ABI data on a geostationary-projection map

    The projection x and y coordinates equals
    the scanning angle (in radians) multiplied by the satellite height
    (http://proj4.org/projections/geos.html)
    """

    scan_date = data_dict['scan_date']
    data = data_dict['data']

    globe = ccrs.Globe(semimajor_axis=data_dict['semimajor_ax'], semiminor_axis=data_dict['semiminor_ax'],
                       flattening=None, inverse_flattening=data_dict['inverse_flattening'])

    Xs, Ys = georeference(data_dict)

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')
    #ax.coastlines(resolution='10m', color='black', linewidth=0.8)

    # TODO: For presentation sample, disable title and add it back in on ppt
    plt.title('GOES-16 Ch.' + str(data_dict['band_id']),
              fontweight='semibold', fontsize=10, loc='left')

    cent_lat = float(center_coords[1])
    cent_lon = float(center_coords[0])

    """
    lim_coords = geodesic_point_buffer(cent_lat, cent_lon, 400)
    lats = [float(x[1]) for x in lim_coords.coords[:]]
    lons = [float(x[0]) for x in lim_coords.coords[:]]

    min_lon = min(lons)
    max_lon = max(lons)

    min_lat = min(lats)
    max_lat = max(lats)

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    """

#    ax.set_extent([lat_lon_extent['w'], lat_lon_extent['e'], lat_lon_extent['s'],
#                   lat_lon_extent['n']], crs=ccrs.PlateCarree())

    band = data_dict['band_id']
    if (band == 11 or band == 13):
        color = cm.binary
    else:
        color = cm.gray

    #color = cm.hsv
    # cmap hsv looks the coolest
    cmesh = plt.pcolormesh(Xs, Ys, data, transform=ccrs.PlateCarree(), cmap=color)

    # Set lat & lon grid tick marks
    lon_ticks = [x for x in range(-180, 181) if x % 2 == 0]
    lat_ticks = [x for x in range(-90, 91) if x % 2 == 0]

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
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=12)
    cbar.set_label('Radiance (' + data_dict['data_units'] + ')', fontsize = 14, labelpad = 20)

    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig(join(out_path, scan_date.strftime('%Y'), scan_date.strftime('%Y%m%d-%H%M')) + '.png', dpi=500)

    #plt.show()
    plt.close(fig)



def plot_abi(data_dict=None, fname=None):
    if (data_dict is None):
        if (fname is None):
            raise ValueError('fname parameter cannot be none')
        else:
            fh = Dataset(fname, mode='r')

            radiance = fh.variables['CMI'][:]
            fh.close()
            fh = None
    else:
        radiance = data_dict['data']

    fig = plt.figure(figsize=(6,6),dpi=200)
    im = plt.imshow(radiance, cmap='Greys')
    cb = fig.colorbar(im, orientation='horizontal')
    cb.set_ticks([1, 100, 200, 300, 400, 500, 600])
    cb.set_label('Radiance (W m-2 sr-1 um-1)')
    plt.show()



def subset_grid(extent, grid_Xs, grid_Ys):
    """
    Finds the ABI grid indexes corresponding to the given min & max lat and lon
    coords

    Parameters
    ----------
    extent : list of float
        List containing the min and max lat & lon coordinates
        Format: min_lat, max_lat, min_lon, max_lon
    grid_Xs : numpy 2D array
    grid_Ys : numpy 2D array

    Returns
    -------
    Tuple of floats
        Indices of the ABI grid corresponding to the min & max lat and lon coords
        Format: (min_y, max_y, min_x, max_x)
    """
    point1 = geod_to_scan(extent[0], extent[2]) # min lat & min lon
    point2 = geod_to_scan(extent[1], extent[3]) # max lat & max lon

    min_x = _find_nearest_idx(grid_Xs, point1[1])
    max_x = _find_nearest_idx(grid_Xs, point2[1])
    min_y = _find_nearest_idx(grid_Ys, point1[0])
    max_y = _find_nearest_idx(grid_Ys, point2[0])

    return (min_y, max_y, min_x, max_x)



def _find_nearest_idx(array, value):
    """
    Helper function called in subset_grid(). Finds the index of the array element
    with the value closest to the parameter value

    Parameters
    ----------
    array : Numpy array
        Array to search for the nearest value
    value : int or float
        Value to search the array for
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
