from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import sys
import re
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

import goesawsinterface
from glm_utils import georeference
from proj_utils import geod_to_scan, scan_to_geod
import plotting_utils


STATES_PATH = '/home/mnichol3/Coding/glm-cases/resources/nws_s_11au16/s_11au16.shp'
COUNTIES_PATH = '/home/mnichol3/Coding/glm-cases/resources/nws_c_02ap19/c_02ap19.shp'


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



def get_abi_files_dict(base_path, satellite, product, t_start, t_end, sector, channel, prompt=False):
    """
    Fetches GOES ABI file(s) for the given satellite, product, sector, & channel,
    for the time period spanning t_start to t_end. Returns a dictionary with the
    scan time as the key and the filename as the value

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
    abi_dict : Dictionary; Key: str, value : str
        Dictionary containing the downloaded ABI file absolute paths.
        Key : ABI file scan time. Format: HHMM
        Value : Absolute path of the downloaded ABI file
    """
    abi_dict = {}
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
            st = datetime.strptime(abi_file.scan_time, '%m-%d-%Y-%H:%M')
            st = datetime.strftime(st, '%H%M')
            abi_dict[st] = abi_file.filepath

    return abi_dict



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
    product_re = r'OR_ABI-L\d\w?-(\w{3,5})\d?-M\d'

    prod_match = re.search(product_re, abi_file)
    if (prod_match):
        prod = prod_match.group(0)
    else:
        raise ValueError('Unable to parse ABI file product')

    fh = Dataset(abi_file, mode='r')

    if ('Rad' in prod):
        prod_key = 'Rad'
        data_dict['min_data_val'] = fh.variables['min_radiance_value_of_valid_pixels'][0]
        data_dict['max_data_val'] = fh.variables['max_radiance_value_of_valid_pixels'][0]
    elif ('CMIP' in prod):
        prod_key = 'CMI'
        data_dict['min_data_val'] = fh.variables['min_brightness_temperature'][0]
        data_dict['max_data_val'] = fh.variables['max_brightness_temperature'][0]
    else:
        raise ValueError('Invalid ABI product key')


    data_dict['band_id'] = fh.variables['band_id'][0]

    data_dict['band_wavelength'] = "%.2f" % fh.variables['band_wavelength'][0]
    data_dict['semimajor_ax'] = fh.variables['goes_imager_projection'].semi_major_axis
    data_dict['semiminor_ax'] = fh.variables['goes_imager_projection'].semi_minor_axis
    data_dict['inverse_flattening'] = fh.variables['goes_imager_projection'].inverse_flattening
    #data_dict['latitude_of_projection_origin'] = fh.variables['goes_imager_projection'].latitude_of_projection_origin
    #data_dict['longitude_of_projection_origin'] = fh.variables['goes_imager_projection'].longitude_of_projection_origin

    data_dict['data_units'] = fh.variables[prod_key].units

    # Seconds since 2000-01-01 12:00:00
    add_seconds = fh.variables['t'][0]

    # Datetime of scan
    scan_date = datetime(2000, 1, 1, 12) + timedelta(seconds=float(add_seconds))

    # Satellite height in meters
    sat_height = fh.variables['goes_imager_projection'].perspective_point_height

    # Satellite longitude & latitude
    sat_lon = fh.variables['goes_imager_projection'].longitude_of_projection_origin
    sat_lat = fh.variables['goes_imager_projection'].latitude_of_projection_origin

    # Satellite lat/lon extend
    lat_lon_extent = {}

    # Geospatial lat/lon center
    data_dict['lat_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lat_center
    data_dict['lon_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lon_center

    # Satellite sweep
    sat_sweep = fh.variables['goes_imager_projection'].sweep_angle_axis

    if (extent is not None):

        # Get the indices of the x & y arrays that define the data subset
        min_y, max_y, min_x, max_x = subset_grid(extent, fh.variables['x'][:], fh.variables['y'][:])

        # Ensure the min & max values are correct
        y_min = min(min_y, max_y)
        y_max = max(min_y, max_y)
        x_min = min(min_x, max_x)
        x_max = max(min_x, max_x)

        data = fh.variables[prod_key][y_min : y_max, x_min : x_max]

        # KEEP!!!!! Determines the plot axis extent
        lat_lon_extent['n'] = extent[1]
        lat_lon_extent['s'] = extent[0]
        lat_lon_extent['e'] = extent[3]
        lat_lon_extent['w'] = extent[2]

        # Y image bounds in scan radians
        # X image bounds in scan radians
        data_dict['y_image_bounds'] = [fh.variables['y'][y_min], fh.variables['y'][y_max]]
        data_dict['x_image_bounds'] = [fh.variables['x'][x_min], fh.variables['x'][x_max]]

    else:
        print('\nWARNING: Not subsetting ABI data!\n')
        data = fh.variables[prod_key][:]

        lat_lon_extent['n'] = fh.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude
        lat_lon_extent['s'] = fh.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
        lat_lon_extent['e'] = fh.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
        lat_lon_extent['w'] = fh.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude

        # Y image bounds in scan radians. Format: (North, South)
        data_dict['y_image_bounds'] = fh.variables['y_image_bounds'][:]

        # X image bounds in scan radians. Format: (west, East)
        data_dict['x_image_bounds'] = fh.variables['x_image_bounds'][:]

    fh.close()
    fh = None

    data_dict['scan_date'] = scan_date
    data_dict['sat_height'] = sat_height
    data_dict['sat_lon'] = sat_lon
    data_dict['sat_lat'] = sat_lat
    data_dict['lat_lon_extent'] = lat_lon_extent
    data_dict['sat_sweep'] = sat_sweep
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

    X, Y = georeference(data_dict['x'], data_dict['y'], sat_lon, sat_height,
                        sat_sweep, data=data_dict['data'])

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(central_longitude=sat_lon,
                                satellite_height=sat_height,false_easting=0,false_northing=0,
                                globe=None, sweep_axis=sat_sweep))


    #ax.set_xlim(int(data_dict['lat_lon_extent']['w']), int(data_dict['lat_lon_extent']['e']))
    #ax.set_ylim(int(data_dict['lat_lon_extent']['s']), int(data_dict['lat_lon_extent']['n']))

    ax.coastlines(resolution='10m', color='gray')
    plt.pcolormesh(X, Y, data_dict['data'], cmap=cm.binary_r, vmin=data_dict['min_data_val'], vmax=data_dict['max_data_val'])

    plt.title('GOES-16 Imagery', fontweight='semibold', fontsize=15)
    plt.title('%s' % scan_date.strftime('%H:%M UTC %d %B %Y'), loc='right')
    ax.axis('equal')

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
    print('Warning: goes_utils.plot_mercator has depricated')
    print('Matts too lazy to modify this function to use imshow')
    sys.exit(0)
    scan_date = data_dict['scan_date']
    data = data_dict['data']

    globe = ccrs.Globe(semimajor_axis=data_dict['semimajor_ax'], semiminor_axis=data_dict['semiminor_ax'],
                       flattening=None, inverse_flattening=data_dict['inverse_flattening'])

    X, Y = georeference(data_dict['x'], data_dict['y'], data_dict['sat_lon'],
                        data_dict['sat_height'], sat_sweep = data_dict['sat_sweep'],
                        data=data_dict['data'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')
    #ax.coastlines(resolution='10m', color='black', linewidth=0.8)

    # TODO: For presentation sample, disable title and add it back in on ppt
    plt.title('GOES-16 Ch.' + str(data_dict['band_id']),
              fontweight='semibold', fontsize=10, loc='left')

    #cent_lat = float(center_coords[1])
    #cent_lon = float(center_coords[0])

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
    cmesh = plt.pcolormesh(X, Y, data, transform=ccrs.PlateCarree(), cmap=color)

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



def get_geospatial_extent(abs_path):
    """
    Gets the geospatial extent of the ABI data in the given file

    Parameters
    ----------
    abs_path : str
        Absolute path of the GOES-16 ABI file to be opened & processed

    Returns
    -------
    extent : dict of floats
        Keys: 'north, south, east, west, lat_center, lon_center'
    """
    extent = {}
    fh = Dataset(abs_path, mode='r')

    extent['north'] = fh.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude
    extent['south'] = fh.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
    extent['east'] = fh.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
    extent['west'] = fh.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude
    extent['lat_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lat_center
    extent['lon_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lon_center
    fh.close() # Not really needed but good practice
    return extent



def plot_sammich_geos(visual, infrared):
    """
    Plots visual & infrared "sandwich" on a geostationary projection. The visual
    imagery provided cloud texture & structure details and the infrared provided
    cloud top temps

    Parameters
    ----------
    visual : dict
        Dictionary of visual satellite data returned by read_file(). Use band 2
    infrared : dict
        Dictionary of infrared satellite data returned by read_file(). Use band 13

    Notes
    -----
    - Uses imshow instead of pcolormesh
    - Passing the Globe object created with ABI metadata to the PlateCarree
      projection causes the shapefiles to not plot properly
    """
    sat_height = visual['sat_height']
    sat_lon = visual['sat_lon']
    sat_sweep = visual['sat_sweep']
    scan_date = visual['scan_date']

    y_min, x_min = scan_to_geod(min(visual['y_image_bounds']), min(visual['x_image_bounds']))
    y_max, x_max = scan_to_geod(max(visual['y_image_bounds']), max(visual['x_image_bounds']))

    globe = ccrs.Globe(semimajor_axis=visual['semimajor_ax'], semiminor_axis=visual['semiminor_ax'],
                       flattening=None, inverse_flattening=visual['inverse_flattening'])

    crs_geos = ccrs.Geostationary(central_longitude=sat_lon,
                                satellite_height=sat_height,false_easting=0,false_northing=0,
                                globe=globe, sweep_axis=sat_sweep)

    trans_pts = crs_geos.transform_points(ccrs.PlateCarree(), np.array([x_min, x_max]), np.array([y_min, y_max]))
    proj_extent = (min(trans_pts[0][0], trans_pts[1][0]), max(trans_pts[0][0], trans_pts[1][0]),
                   min(trans_pts[0][1], trans_pts[1][1]), max(trans_pts[0][1], trans_pts[1][1]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=crs_geos)

    states = shpreader.Reader(STATES_PATH)
    states = list(states.geometries())
    states = cfeature.ShapelyFeature(states, ccrs.PlateCarree())

    counties = shpreader.Reader(COUNTIES_PATH)
    counties = list(counties.geometries())
    counties = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    #ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(states, linewidth=.8, facecolor='none', edgecolor='gray', zorder=3)
    ax.add_feature(counties, linewidth=.3, facecolor='none', edgecolor='gray', zorder=3)

    # visual & infrared arrays are different dimensions
    # viz_img = plt.imshow(visual['data'], cmap=cm.binary_r, extent=extent,
    #                      vmin=visual['min_data_val'], vmax=visual['max_data_val'], zorder=1)
    viz_img = plt.imshow(visual['data'], cmap=cm.binary_r, vmin=visual['min_data_val'],
                         vmax=visual['max_data_val'], zorder=1, transform=crs_geos, extent=proj_extent)

    infrared_norm = colors.LogNorm(vmin=190, vmax=270)
    # inf_img = plt.imshow(infrared['data'], cmap=cm.nipy_spectral_r, norm=infrared_norm,
    #            extent=extent, zorder=2, alpha=0.4)
    inf_img = plt.imshow(infrared['data'], cmap=cm.nipy_spectral_r, norm=infrared_norm,
                         zorder=2, alpha=0.4, transform=crs_geos, extent=proj_extent)

    cbar_bounds = np.arange(190, 270, 10)
    cbar = plt.colorbar(inf_img, ticks=cbar_bounds, spacing='proportional')
    cbar.ax.set_yticklabels([str(x) for x in cbar_bounds])

    plt.title('GOES-16 Imagery', fontweight='semibold', fontsize=15)
    plt.title('%s' % scan_date.strftime('%H:%M UTC %d %B %Y'), loc='right')
    #ax.axis('equal')  # May cause shapefile to extent beyond borders of figure

    plt.show()



def plot_sammich_mercator(visual, infrared):
    """
    Plots visual & infrared "sandwich" on a Mercator projection. The visual
    imagery provided cloud texture & structure details and the infrared provided
    cloud top temps

    Parameters
    ----------
    visual : dict
        Dictionary of visual satellite data returned by read_file(). Use band 2
    infrared : dict
        Dictionary of infrared satellite data returned by read_file(). Use band 13

    Notes
    -----
    - Uses imshow instead of pcolormesh
    - Passing the Globe object created with ABI metadata to the PlateCarree
      projection causes the shapefiles to not plot properly
    """
    sat_height = visual['sat_height']
    sat_lon = visual['sat_lon']
    sat_sweep = visual['sat_sweep']
    scan_date = visual['scan_date']

    # Left, Right, Bottom, Top
    extent = [visual['lat_lon_extent']['w'], visual['lat_lon_extent']['e'],
              visual['lat_lon_extent']['s'], visual['lat_lon_extent']['n']]

    y_min, x_min = scan_to_geod(min(visual['y_image_bounds']), min(visual['x_image_bounds']))
    y_max, x_max = scan_to_geod(max(visual['y_image_bounds']), max(visual['x_image_bounds']))

    globe = ccrs.Globe(semimajor_axis=visual['semimajor_ax'], semiminor_axis=visual['semiminor_ax'],
                       flattening=None, inverse_flattening=visual['inverse_flattening'])

    crs_geos = ccrs.Geostationary(central_longitude=sat_lon, satellite_height=sat_height,
                                   false_easting=0, false_northing=0, globe=globe, sweep_axis=sat_sweep)

    crs_plt = ccrs.PlateCarree() # Globe keyword was messing everything up

    trans_pts = crs_geos.transform_points(crs_plt, np.array([x_min, x_max]), np.array([y_min, y_max]))

    proj_extent = (min(trans_pts[0][0], trans_pts[1][0]), max(trans_pts[0][0], trans_pts[1][0]),
                   min(trans_pts[0][1], trans_pts[1][1]), max(trans_pts[0][1], trans_pts[1][1]))

    ##################### Filter WWA polygons ######################
    # print('Processing state shapefiles...\n')
    # polys = _filter_polys(STATES_PATH, extent)
    # states = cfeature.ShapelyFeature(polys, ccrs.PlateCarree())
    #
    # print('Processing county shapefiles...\n')
    # polys = _filter_polys(COUNTIES_PATH, extent)
    # counties = cfeature.ShapelyFeature(polys, ccrs.PlateCarree())

    print('\nProcessing state shapefiles...\n')
    states = shpreader.Reader(STATES_PATH)
    states = list(states.geometries())
    states = cfeature.ShapelyFeature(states, crs_plt)

    print('Processing county shapefiles...\n')
    counties = shpreader.Reader(COUNTIES_PATH)
    counties = list(counties.geometries())
    counties = cfeature.ShapelyFeature(counties, crs_plt)
    ################################################################

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent(extent, crs=crs_plt)

    print('Creating map...\n')

    ax.add_feature(states, linewidth=.3, facecolor='none', edgecolor='black', zorder=3)
    ax.add_feature(counties, linewidth=.1, facecolor='none', edgecolor='black', zorder=3)

    viz_img = plt.imshow(visual['data'], cmap=cm.Greys_r, extent=proj_extent, origin='upper',
                         vmin=visual['min_data_val'], vmax=visual['max_data_val'],
                         zorder=1, transform=crs_geos, interpolation='none')

    infrared_norm = colors.LogNorm(vmin=190, vmax=270)
    custom_cmap = plotting_utils.custom_cmap()
    inf_img = plt.imshow(infrared['data'], cmap=custom_cmap, extent=proj_extent, origin='upper',
                         norm=infrared_norm, zorder=2, alpha=0.4, transform=crs_geos, interpolation='none')

    cbar_bounds = np.arange(190, 270, 10)
    cbar = plt.colorbar(inf_img, ticks=cbar_bounds, spacing='proportional')
    cbar.ax.set_yticklabels([str(x) for x in cbar_bounds])

    plt.title('GOES-16 Imagery', fontweight='semibold', fontsize=15)
    plt.title('%s' % scan_date.strftime('%H:%M UTC %d %B %Y'), loc='right')
    #ax.axis('equal')  # May cause shapefile to extent beyond borders of figure

    plt.show()



def _filter_polys(shp_fname, extent):
    w = extent[0]
    e = extent[1]
    s = extent[2]
    n = extent[3]
    filtered_polys = []
    view_bbox = Polygon([(w, s), (e, s), (e, n), (w, n)])

    shp_poly = shpreader.Reader(shp_fname)
    for rec in shp_poly.records():
        bounds = rec.geometry.bounds  # (minx, miny, maxx, maxy)
        # Make sure the lat & lon signs are correct, many cases they aren't
        if (np.sign(w) != np.sign(bounds[0])):
            bound_w = -1 * bounds[0]
        else:
            bound_w = bounds[0]
        if (np.sign(e) != np.sign(bounds[2])):
            bound_e = -1 * bounds[2]
        else:
            bound_e = bounds[2]
        if (np.sign(s) != np.sign(bounds[1])):
            bound_s = -1 * bounds[1]
        else:
            bound_s = bounds[1]
        if (np.sign(n) != np.sign(bounds[3])):
            bound_n = -1 * bounds[3]
        else:
            bound_n = bounds[3]

        points = [Point(bound_w, bound_s), Point(bound_e, bound_s),
                  Point(bound_e, bound_n), Point(bound_w, bound_n)]

        for p in points:
            if (view_bbox.contains(p)):
                filtered_polys.append(rec.geometry)
                #yield rec.geometry
                break
    return filtered_polys





def _rad_to_ref(radiance, channel=2, correct=True):
    """
    Performs a linear conversion of spectral radiance to reflectance factor

    Parameters
    ----------
    radiance : numpy 2d array
        2D array of radiance values
        Units: mW / (m**2 sr cm**-1)
    correct : bool, optional
        If True, the reflectance array is gamma-corrected
    channel : int, optional
        ABI channel pertaining to the radiance data. Default: 2

    Returns
    -------
    ref : numpy 2D array
        2D array of reflectance values with the same dimensions as 'radiance'
    """
    constants = {1: 726.721072, 2: 663.274497, 3: 441.868715}
    d2 = 0.3

    if (channel not in constants.keys()):
        raise ValueError('Invalid ABI channel')

    ref = (radiance * np.pi / d2) / constants[channel]

    # Ensure the data is nominal
    ref = np.maximum(ref, 0.0)
    ref = np.minimum(ref, 1.0)

    if (correct):
        ref = _gamma_corr(ref)

    return ref



def _gamma_corr(ref):
    """
    Adjusts the reflectance array. Results in a brighter image when plotted

    Parameters
    ----------
    ref : numpy 2d array
        2D array of reflectance

    Returns
    -------
    gamma_ref : numpy 2d array
        2D array of gamma-corrected reflectance values
    """
    gamma_ref = np.sqrt(ref)
    return gamma_ref



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
