import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.ndimage
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.ndimage
import re

from glm_utils import georeference
import grib
from plotting_utils import to_file, load_data, load_coordinates, parse_coord_fnames,
                           process_slice, process_slice_inset, 



def plot_mercator_dual(glm_obj, extent_coords, wtlma_obj):

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    ext_lats = [extent_coords[0][0], extent_coords[1][0]]
    ext_lons = [extent_coords[0][1], extent_coords[1][1]]

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp', zorder=0)

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    ax.set_extent([min(ext_lons), max(ext_lons), min(ext_lats), max(ext_lats)], crs=ccrs.PlateCarree())

    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = plt.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=ccrs.PlateCarree(), cmap=cm.jet, zorder=2)

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional', fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.set_label('GLM Flash Extent Density')

    scat = plt.scatter(wtlma_obj.data['lon'], wtlma_obj.data['lat'], c=wtlma_obj.data['P'],
                       marker='o', s=100, cmap=cm.gist_ncar_r, vmin=-20, vmax=100, zorder=3, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(scat, fraction=0.046, pad=0.04)
    cbar2.set_label('WTLMA Flash Power (dBW)')

    plt.title('GLM FED {} {}\n WTLMA Flashes {}'.format(glm_obj.scan_date, glm_obj.scan_time, wtlma_obj._start_time_pp()), loc='right')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def plot_mercator_dual_2(glm_obj, extent_coords, wtlma_obj):
    """
    Same as plot_mercator_dual(), except it plots the wtlma strokes as
    power densities
    """
    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    ext_lats = [extent_coords[0][0], extent_coords[1][0]]
    ext_lons = [extent_coords[0][1], extent_coords[1][1]]

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp', zorder=0)

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    ax.set_extent([min(ext_lons), max(ext_lons), min(ext_lats), max(ext_lats)], crs=ccrs.PlateCarree())


    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = plt.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=ccrs.PlateCarree(), cmap=cm.jet, zorder=3, alpha=0.5)

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional', fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.set_label('GLM Flash Extent Density')


    grid_lons = np.arange(min(ext_lons), max(ext_lons), 0.01)
    grid_lats = np.arange(min(ext_lats), max(ext_lats), 0.01)

    lma_norm = colors.LogNorm(vmin=1, vmax=150)

    H, X_edges, Y_edges = np.histogram2d(wtlma_obj.data['lon'], wtlma_obj.data['lat'],
                          bins=250, range=[[min(ext_lons), max(ext_lons)], [min(ext_lats), max(ext_lats)]],
                          weights=wtlma_obj.data['P']) # bins=[len(grid_lons), len(grid_lats)]

    lma_mesh = plt.pcolormesh(X_edges, Y_edges, H.T, norm=lma_norm, transform=ccrs.PlateCarree(), cmap=cm.inferno, zorder=2)

    lma_bounds = [5, 10, 15, 20, 25, 50, 100, 150]
    cbar2 = plt.colorbar(lma_mesh, ticks=lma_bounds, spacing='proportional',fraction=0.046, pad=0.04)
    cbar2.ax.set_yticklabels([str(x) for x in lma_bounds])
    cbar2.set_label('WTLMA Flash Power Density (dBW)')

    plt.title('GLM FED {} {}\n WTLMA Flashes {}'.format(glm_obj.scan_date, glm_obj.scan_time, wtlma_obj._start_time_pp()), loc='right')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def plot_mercator_glm_subset(data_dict, extent_coords):

    y1, x1 = geod_to_scan(point1[0], point1[1])
    y2, x2 = geod_to_scan(point2[0], point2[1])

    xs = np.asarray(data['x'])
    ix_1 = (np.abs(xs - x1)).argmin()
    ix_2 = (np.abs(xs - x2)).argmin()
    x_subs = xs[min(ix_1, ix_2) : max(ix_1, ix_2)]

    ys = np.asarray(data['y'])
    iy_1 = (np.abs(ys - y1)).argmin()
    iy_2 = (np.abs(ys - y2)).argmin()
    y_subs = ys[min(iy_1, iy_2):max(iy_1, iy_2)]

    globe = ccrs.Globe(semimajor_axis=data['semi_major_axis'], semiminor_axis=data['semi_minor_axis'],
                       flattening=None, inverse_flattening=data['inv_flattening'])

    lons, lats = georeference(x_subs, y_subs, data['lon_0'], data['height'], data['sweep_ang_axis'])

    data_subs = data['data'][476:576, 715:760]

    #print('Size of data_subs (bytes):', data_subs.nbytes)
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    ax.set_extent([-102.185, -99.865, 34.565, 37.195], crs=ccrs.PlateCarree())

    cmesh = plt.pcolormesh(lons, lats, data_subs, vmin=0, vmax=350, transform=ccrs.PlateCarree(), cmap=cm.jet)
    cbar = plt.colorbar(cmesh,fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def plot_cross_cubic_single(grb, point1, point2, first=False):
    """
    Plots the cross section of a single MRMSGrib object's data from point1 to point2
    using cubic interpolation

    Parameters
    ----------
    grb : MRMSGrib object
    point1 : tuple of float
        Coordinates of the first point that defined the cross section
        Format: (lon, lat)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lon, lat)
    first : bool, optional
        If True, the cross section latitude & longitude coordinates will be calculated
        and written to text files

    Returns
    -------
    None, displays a plot of the cross section

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
    x_world, y_world = np.array(list(zip(*line)))
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y.max() - y_world ) / y.ptp()

    num = 1000
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    valid_date = grb.validity_date
    valid_time = grb.validity_time

    if (first):

        fname_lons = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lons.txt'
        fname_lats = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lats.txt'

        d_lons, d_lats = calc_coords(point1, point2, num)

        to_file(BASE_PATH_XSECT + '/coords', fname_lons, d_lons)
        to_file(BASE_PATH_XSECT + '/coords', fname_lats, d_lats)

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(z, np.vstack((row, col)), order=1, mode='nearest')

    # Plot...
    fig, axes = plt.subplots(nrows=2)
    axes[0].pcolormesh(x, y, z)
    axes[0].plot(x_world, y_world, 'ro-')
    axes[0].axis('image')

    axes[1].plot(zi)

    plt.show()



def plot_cross_neighbor_single(grb, point1, point2, first=False):
    """
    Plots the cross section of a single MRMSGrib object's data from point1 to point2
    using nearest-neighbor interpolation

    Parameters
    ----------
    grb : MRMSGrib object
    point1 : tuple of float
        Coordinates of the first point that defined the cross section
        Format: (lon, lat)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lon, lat)
    first : bool, optional
        If True, the cross section latitude & longitude coordinates will be calculated
        and written to text files

    Returns
    -------
    None, displays a plot of the cross section
    """
    BASE_PATH = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    BASE_PATH_XSECT = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'
    BASE_PATH_XSECT_COORDS = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/coords'

    lons = grb.grid_lons
    lats = grb.grid_lats

    x, y = np.meshgrid(lons, lats)
    z = grb.data

    line = [(point1[0], point1[1]), (point2[0], point2[1])]
    x_world, y_world = np.array(list(zip(*line)))
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y.max() - y_world ) / y.ptp()

    num = 1000
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    valid_date = grb.validity_date
    valid_time = grb.validity_time

    if (first):

        fname_lons = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lons.txt'
        fname_lats = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lats.txt'

        d_lons, d_lats = calc_coords(point1, point2, num)

        to_file(BASE_PATH_XSECT + '/coords', fname_lons, d_lons)
        to_file(BASE_PATH_XSECT + '/coords', fname_lats, d_lats)

    zi = z[row.astype(int), col.astype(int)] #(10000,)

    fig, axes = plt.subplots(nrows=2)
    axes[0].pcolormesh(x, y, z)
    axes[0].plot(x_world, y_world, 'ro-')
    axes[0].axis('image')

    axes[1].plot(zi)

    plt.show()



def get_cross_cubic(grb, point1, point2, first=False):
    """
    Calculates the cross section of a single MRMSGrib object's data from point1 to point2
    using cubic interpolation

    Parameters
    ----------
    grb : MRMSGrib object
    point1 : tuple of float
        Coordinates of the first point that defined the cross section
        Format: (lon, lat)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lon, lat)
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
    x_world, y_world = np.array(list(zip(*line)))
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y.max() - y_world ) / y.ptp()

    num = 100
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    valid_date = grb.validity_date
    valid_time = grb.validity_time

    if (first):

        fname_lons = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lons.txt'
        fname_lats = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lats.txt'

        d_lons, d_lats = calc_coords(point1, point2, num)

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
        Format: (lon, lat)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lon, lat)
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

    line = [(point1[0], point1[1]), (point2[0], point2[1])]
    x_world, y_world = np.array(list(zip(*line)))
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y.max() - y_world ) / y.ptp()

    num = 1000
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    valid_date = grb.validity_date
    valid_time = grb.validity_time

    if (first):

        fname_lons = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lons.txt'
        fname_lats = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z-lats.txt'

        d_lons, d_lats = calc_coords(point1, point2, num)

        to_file(BASE_PATH_XSECT + '/coords', fname_lons, d_lons)
        to_file(BASE_PATH_XSECT + '/coords', fname_lats, d_lats)

    zi = z[row.astype(int), col.astype(int)]

    return zi



def plot_mrms_cross_section(data=None, abs_path=None, lons=None, lats=None):
    """
    Plots a cross-section of MRMS reflectivity data from all scan angles. If
    the 'data' parameter is given, then that data is plotted. If 'abs_path' is
    given, then data from the text file located at that absolute path is read and
    plotted

    Parameters
    ----------
    data : numpy 2d array, optional
        2d array of reflectivity data
    abs_path : str, optional
        Absolute path of the text file containing the reflectivity cross-section
        data. Must be given if data is None
    lons : list of float, optional
        List of longitude coordinates from the vertical slice. Must be given if
        cross section data is passed in through the data parameter
    lats : list of float, optional
        List of latitude coordinates from the vertical slice. Must be given if
        cross section data is passed in through the data parameter

    Returns
    -------
    None, displays a plot of the reflectivity cross section
    """

    scan_angles = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
                            3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    if (data is not None):
        if (lons is None or lats is None):
            raise ValueError('lons and/or lats parameters cannot be None')
        else:
            coords = list(zip(lons, lats))
    else:
        if (abs_path is None):
            raise ValueError('data and abs_path parameters cannot both be None')
        else:
            data = load_data(abs_path)
            f_lon, f_lat = parse_coord_fnames(abs_path)
            lons = load_coordinates(f_lon)
            lats = load_coordinates(f_lat)

            coords = []
            for idx, x in enumerate(lons):
                coords.append(str(x) + ', ' + str(lats[idx]))

    fig = plt.figure()
    ax = plt.gca()

    xs = np.arange(0, 1000)

    #im = ax.pcolormesh(xs, scan_angles, data, cmap=mpl.cm.gist_ncar)
    im = ax.pcolormesh(coords, scan_angles, data, cmap=mpl.cm.gist_ncar)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Reflectivity (dbz)', rotation=90)
    ax.set_title('MRMS Reflectivity Cross Section')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_ylabel('Scan Angle (Deg)')
    ax.set_xlabel('Lon, Lat')

    fig.tight_layout()

    plt.show()



def plot_mrms_cross_section_inset(data=None, inset_data=None, inset_lons=None, inset_lats=None, abs_path=None, lons=None, lats=None, points=None):
    """
    Plots a cross-section of MRMS reflectivity data from all scan angles. If
    the 'data' parameter is given, then that data is plotted. If 'abs_path' is
    given, then data from the text file located at that absolute path is read and
    plotted

    Parameters
    ----------
    data : numpy 2d array, optional
        2d array of reflectivity data
    abs_path : str, optional
        Absolute path of the text file containing the reflectivity cross-section
        data. Must be given if data is None
    lons : list of float, optional
        List of longitude coordinates from the vertical slice. Must be given if
        cross section data is passed in through the data parameter
    lats : list of float, optional
        List of latitude coordinates from the vertical slice. Must be given if
        cross section data is passed in through the data parameter

    Returns
    -------
    None, displays a plot of the reflectivity cross section
    """
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from cartopy.feature import NaturalEarthFeature

    scan_angles = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
                            3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    if (inset_data is None or inset_lons is None or inset_lats is None or points is None):
        raise ValueError('Missing inset data')
    if (data is not None):
        if (lons is None or lats is None):
            raise ValueError('lons and/or lats parameters cannot be None')
        else:
            coords = list(zip(lons, lats))
    else:
        if (abs_path is None):
            raise ValueError('data and abs_path parameters cannot both be None')
        else:
            data = load_data(abs_path)
            f_lon, f_lat = parse_coord_fnames(abs_path)
            lons = load_coordinates(f_lon)
            lats = load_coordinates(f_lat)
            inset_data = load_data(inset_data)
            inset_lons = load_data(inset_lons)
            inset_lats = load_data(inset_lats)

            coords = []
            for idx, x in enumerate(lons):
                coords.append(str(x) + ', ' + str(lats[idx]))

    fig = plt.figure()
    ax = plt.gca()

    xs = np.arange(0, 1000)

    #im = ax.pcolormesh(xs, scan_angles, data, cmap=mpl.cm.gist_ncar)
    im = ax.pcolormesh(coords, scan_angles, data, cmap=mpl.cm.gist_ncar)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Reflectivity (dbz)', rotation=90)
    ax.set_title('MRMS Reflectivity Cross Section')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_ylabel('Scan Angle (Deg)')
    ax.set_xlabel('Lon, Lat')

    # pelson.github.io/cartopy/examples/effects_of_the_ellipse.html
    sub_ax = plt.axes([0.07, 0.7, .25, .25], projection=ccrs.PlateCarree(), facecolor='w')
    sub_ax.set_extent([min(lons)-0.05, max(lons)+0.05, min(lats)-0.05, max(lats)+0.05], crs=ccrs.PlateCarree())

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    sub_ax.add_feature(states, linewidth=.8, edgecolor='black')

    cmesh = plt.pcolormesh(inset_lons, inset_lats, inset_data, transform=ccrs.PlateCarree(), cmap=cm.gist_ncar)
    xs = [points[0][0], points[1][0]]
    ys = [points[0][1], points[1][1]]

    sub_ax.plot(xs, ys, 'ro-', transform=ccrs.PlateCarree())
    fig.tight_layout()

    plt.show()


"""
def run(base_path, slice_time, point1, point2):
    fname = process_slice(base_path, slice_time, point1, point2, write=True)
    plot_cross_section(abs_path=fname)



def run_inset(base_path, slice_time, point1, point2):
    f_dict = process_slice_inset(base_path, '2124', point1, point2)
    plot_cross_section_inset(inset_data=f_dict['f_inset_data'], inset_lons=f_dict['f_inset_lons'],
                             inset_lats=f_dict['f_inset_lats'], abs_path=f_dict['x_sect'], points=(point1, point2))
"""
