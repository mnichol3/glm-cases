from grib import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from os.path import join, isdir, isfile
import scipy.ndimage
import tracemalloc
from os import mkdir
import matplotlib as mpl




def plot_cross_cubic_single(grb, point1, point2):
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

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(z, np.vstack((row, col)), order=1, mode='nearest')

    # Plot...
    fig, axes = plt.subplots(nrows=2)
    axes[0].pcolormesh(x, y, z)
    axes[0].plot(x_world, y_world, 'ro-')
    axes[0].axis('image')

    axes[1].plot(zi)

    plt.show()



def plot_cross_neighbor_single(grb, point1, point2):
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
    zi = z[row.astype(int), col.astype(int)] #(10000,)

    fig, axes = plt.subplots(nrows=2)
    axes[0].pcolormesh(x, y, z)
    axes[0].plot(x_world, y_world, 'ro-')
    axes[0].axis('image')

    axes[1].plot(zi)

    plt.show()



def get_cross_cubic(grb, point1, point2):
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

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(z, np.vstack((row, col)), order=1, mode='nearest')

    return zi



def get_cross_neighbor(grb, point1, point2):
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

    zi = z[row.astype(int), col.astype(int)] #(10000,)

    return zi



def plot_cross_section(data=None, abs_path=None):
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

    Returns
    -------
    None, displays a plot of the reflectivity cross section
    """

    if (data is None):
        if (abs_path is None):
            raise ValueError('data and abs_path parameters cannot both be None')
        else:
            data = load_data(abs_path)

    scan_angles = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
                            3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    fig = plt.figure()
    ax = plt.gca()

    xs = np.arange(0, 1000)

    im = ax.pcolormesh(xs, scan_angles, data, cmap=mpl.cm.gist_ncar)
    fig.colorbar(im, ax=ax)
    ax.set_title('MRMS Reflectivity Cross Section')

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
    None
    """

    if (not isdir(out_path)):
        mkdir(out_path)

    abs_path = join(out_path, f_name)

    print("\nWriting", abs_path, "\n")

    np.savetxt(abs_path, data, delimiter=',', newline='\n', fmt='%2.2f')



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
        print('Loading MRMS cross_section data from', abs_path)
        data = np.loadtxt(abs_path, dtype=float, delimiter=',')

        return data



def main():

    tracemalloc.start()
    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms'
    #f_name = 'MRMS_MergedReflectivityQC_00.50_20190523-212434.grib2'
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905/MergedReflectivityQC_01.50'
    #f_name = 'MRMS_MergedReflectivityQC_01.50_20190523-212434.grib2'
    #f_abs = join(f_path, f_name)
    f_out = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'

    """
    scans = fetch_scans(base_path, '2124') # z = 33

    grib_files = get_grib_objs(scans, base_path)
    print(grib_files)

    data = [x.data for x in grib_files]
    """
    """
    files = ['MRMS_MergedReflectivityQC_02.00_20190523-212434.grib2',
             'MRMS_MergedReflectivityQC_02.25_20190523-212434.grib2']
    """

    """
    cross_sections = np.array([])

    #(-101.822, 35.0833), (-100.403, 37.1292)
    point1 = (-101.618, 35.3263)
    point2 = (-100.999, 36.2826)

    scans = fetch_scans(base_path, '2124') # z = 33

    grbs = get_grib_objs(scans, base_path)

    valid_date = grbs[0].validity_date
    valid_time = grbs[0].validity_time

    fname = 'mrms-cross-' + str(valid_date) + '-' + str(valid_time) + 'z.txt'

    cross_sections = np.asarray(get_cross_neighbor(grbs[0], point1, point2))

    for grb in grbs[1:]:
        cross_sections = np.vstack((cross_sections, get_cross_neighbor(grb, point1, point2)))

    to_file(f_out, fname, cross_sections)

    print("Memory Useage - Current: %d, Peak: %d" % tracemalloc.get_traced_memory())
    """
    fname = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect/mrms-cross-20190523-2124z.txt'

    plot_cross_section(abs_path=fname)





if (__name__ == '__main__'):
    main()
