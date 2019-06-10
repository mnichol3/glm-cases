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



def read_file(abi_file):
    """
    Opens & reads a GOES-16 ABI data file, returning a dictionary of data

    !!! NOTE: Returns implroper sat_lon value; return 75.0 but should be 75.2 for
    GOES-16

    Parameters:
    ------------
    fname : str
        Name of the GOES-16 ABI date file to be opened & processed


    Returns:
    ------------
    data_dict : dictionary of str
        Dictionar of ABI image data & metadata from the netCDF file
    """

    data_dict = {}

    # Ch. 1 - Blue vis. Good resolution, not able to use at night

    # Ch. 11 - Cloud top infrared. Has much lower resolution than visible bands

    # Ch. 13 - 'Clean' Longwave window. Not much different than Ch. 11


    fh = Dataset(join(PATH_LINUX_ABI, abi_file), mode='r')

    data_dict['band_id'] = fh.variables['band_id'][0]

    if (data_dict['band_id'] < 8):
        print('\n!!! WARNING: Currently plotting non-IR satellite data !!!' )

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

    data = fh.variables['CMI'][:].data

    Xs = fh.variables['x'][:]
    Ys = fh.variables['y'][:]

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



def georeference(data_dict):
    """
    Calculates the longitude and latitude coordinates of each data point

    Parameters
    ------------
    data_dict : dictionary
        Dictionary of ABI file data & metadata


    Returns
    ------------
    (lons, lats) : tuple of lists of floats
        Tuple containing a list of data longitude coordinates and a list of
        data latitude coordinates
    """

    sat_height = data_dict['sat_height']
    sat_lon = data_dict['sat_lon']
    sat_sweep = data_dict['sat_sweep']
    data = data_dict['data'] # (1000, 1000) array

    # Multiplying by sat height might not be necessary here
    Xs = data_dict['x'] * sat_height # (1000,)
    Ys = data_dict['y'] * sat_height # (1000,)

    p = pyproj.Proj(proj='geos', h=sat_height, lon_0=sat_lon, sweep=sat_sweep)

    lons, lats = np.meshgrid(Xs, Ys)
    lons, lats = p(lons, lats, inverse=True)

    lats[np.isnan(data)] = 57
    lons[np.isnan(data)] = -152

    return (lons, lats)



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



def plot_abi(fname):
    fh = Dataset(fname, mode='r')

    radiance = fh.variables['CMI'][:]
    fh.close()
    fh = None

    fig = plt.figure(figsize=(6,6),dpi=200)
    im = plt.imshow(radiance, cmap='Greys')
    cb = fig.colorbar(im, orientation='horizontal')
    cb.set_ticks([1, 100, 200, 300, 400, 500, 600])
    cb.set_label('Radiance (W m-2 sr-1 um-1)')
    plt.show()