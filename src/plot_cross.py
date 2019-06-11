from grib import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from os.path import join
import scipy.ndimage



def plot_cross_cubic(grb, point1, point2):
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



def plot_cross_neighbor(grb, point1, point2):
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



def main():

    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms'
    #f_name = 'MRMS_MergedReflectivityQC_00.50_20190523-212434.grib2'
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905/MergedReflectivityQC_01.50'
    #f_name = 'MRMS_MergedReflectivityQC_01.50_20190523-212434.grib2'
    #f_abs = join(f_path, f_name)

    """
    scans = fetch_scans(base_path, '2124') # z = 33

    grib_files = get_grib_objs(scans, base_path)
    print(grib_files)

    data = [x.data for x in grib_files]
    """

    files = ['MRMS_MergedReflectivityQC_02.00_20190523-212434.grib2',
             'MRMS_MergedReflectivityQC_02.25_20190523-212434.grib2']

    #(-101.822, 35.0833), (-100.403, 37.1292)
    point1 = (-101.618, 35.3263)
    point2 = (-100.999, 36.2826)


    grbs = get_grib_objs(files, base_path)
    get_cross_neighbor(grbs[0], point1, point2)






if (__name__ == '__main__'):
    main()
