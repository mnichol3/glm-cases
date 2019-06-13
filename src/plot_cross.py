from grib import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from os.path import join, isdir, isfile
import scipy.ndimage
import tracemalloc
from os import mkdir
import matplotlib as mpl
import re


BASE_PATH = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
BASE_PATH_XSECT = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'
BASE_PATH_XSECT_COORDS = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/coords'


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



def calc_coords(point1, point2, num):
    """
    Calculates the longitude and latitude coordinates along the cross-section
    slice

    Parameters
    ----------
    point1 : tuple of float
        First point defining the cross section
        Format: (lon, lat)
    point2 : tuple of float
        Second point defining the cross section
        Format: (lon, lat)
    num : int
        Number of coordinate points to calculate

    Returns
    -------
    tuple of lists of float
        Tuple containing two list; the first containing longitude coordinates,
        the second containing latitude coordinates
        Format: (lons, lats)
    """
    lons = np.linspace(point1[0], point2[0], num, endpoint=True, dtype=np.double)
    lons = trunc(lons, 3)

    lats = np.linspace(point1[1], point2[1], num, endpoint=True, dtype=np.double)
    lats = trunc(lats, 3)

    return (lons, lats)



def plot_cross_section(data=None, abs_path=None, lons=None, lats=None):
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



def plot_cross_section_inset(data=None, inset_data=None, inset_lons=None, inset_lats=None, abs_path=None, lons=None, lats=None, points=None):
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

    scans = fetch_scans(BASE_PATH, slice_time) # z = 33

    grbs = get_grib_objs(scans, BASE_PATH)

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
    tracemalloc.start()
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



def run(base_path, slice_time, point1, point2):
    fname = process_slice(base_path, slice_time, point1, point2, write=True)
    plot_cross_section(abs_path=fname)



def run_inset(base_path, slice_time, point1, point2):
    f_dict = process_slice_inset(base_path, '2124', point1, point2)
    plot_cross_section_inset(inset_data=f_dict['f_inset_data'], inset_lons=f_dict['f_inset_lons'],
                             inset_lats=f_dict['f_inset_lats'], abs_path=f_dict['x_sect'], points=(point1, point2))



def main():

    tracemalloc.start()
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    f_out = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'


    point1 = (-101.618, 35.3263)
    point2 = (-100.999, 36.2826)

    # Plot cross section without inset
    # run(base_path, '2124', point1, point2)

    # Plot cross section with inset
    #run_inset(base_path, '2124', point1, point2)

    print("Memory Useage - Current: %d, Peak: %d" % tracemalloc.get_traced_memory())



if (__name__ == '__main__'):
    main()
