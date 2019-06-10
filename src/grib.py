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
from os import listdir
from gribobj import GribObj


def print_keys(fname, keyword=None):
    grb = pygrib.open(fname)
    grb = grb[1]

    if (keyword is not None):
        for key in grb.keys():
            if keyword in key:
                print(key)
    else:
        for key in grb.keys():
            print(key)
    print('------------------------------------')



def get_keys(fname, keyword=None):
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



def get_grb_data(fname, debug=False):
    grb = pygrib.open(fname)
    grb = grb[1]

    data = grb.values
    major_ax = grb.earthMajorAxis
    minor_ax = grb.earthMinorAxis
    val_date = grb.validityDate
    val_time = grb.validityTime

    #data[data <= 0] = float('nan')

    if (debug):
        print('data array shape (y, x):', data.shape)
        print('validity date:', val_date)
        print('validity time:', val_time)
        print('major axis:', major_ax)
        print('minor axis:', minor_ax)
        print('------------------------------------')

    return GribObj(val_date, val_time, data, major_ax, minor_ax)



def plot_grb(fname, grid=None):

    grb = pygrib.open(fname)
    grb = grb[1]

    major_ax = grb.earthMajorAxis
    minor_ax = grb.earthMinorAxis

    if (grid):
        lon = grid[0]
        lat = grid[1]
    else:
        lat, lon = grb.latlons()

    data = grb.values

    data[data <= 0] = float('nan')


    fig = plt.figure(figsize=(10, 5))

    globe = ccrs.Globe(semimajor_axis=major_ax, semiminor_axis=minor_ax,
                       flattening=None)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    cmesh = plt.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cm.gist_ncar)

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
    plt.title('MRMS Reflectivity ' + str(grb.validityDate) + ' ' + str(grb.validityTime) + 'z')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=12)
    cbar.set_label('Reflectivity (dbz)', fontsize = 14, labelpad = 20)

    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    #fig.savefig(join(out_path, scan_date.strftime('%Y'), scan_date.strftime('%Y%m%d-%H%M')) + '.png', dpi=500)

    plt.show()



def plot_grb_data(grb):

    fig = plt.figure(figsize=(10, 5))

    globe = ccrs.Globe(semimajor_axis=grb.major_axis, semiminor_axis=grb.minor_axis,
                       flattening=None)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    cmesh = plt.pcolormesh(grb.grid_lons, grb.grid_lats, grb.data, transform=ccrs.PlateCarree(), cmap=cm.gist_ncar)

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
    plt.title('MRMS Reflectivity ' + str(grb.validity_date) + ' ' + str(grb.validity_time) + 'z')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=12)
    cbar.set_label('Reflectivity (dbz)', fontsize = 14, labelpad = 20)

    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    #fig.savefig(join(out_path, scan_date.strftime('%Y'), scan_date.strftime('%Y%m%d-%H%M')) + '.png', dpi=500)

    plt.show()



def get_files_in_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    return files



def grid_info(fname):
    grb = pygrib.open(fname)
    grb = grb[1]

    print('Grid type:', grb.gridType)
    print('Grid Description Section Present:', grb.gridDescriptionSectionPresent)
    print('Grid Definition Template Number:', grb.gridDefinitionTemplateNumber)
    print('Grid Definition Description:', grb.gridDefinitionDescription)
    print('Grid Longitudes (First, Last):', grb.longitudeOfFirstGridPointInDegrees, grb.longitudeOfLastGridPointInDegrees)
    print('Grid Latitudes (First, Last):', grb.latitudeOfFirstGridPointInDegrees, grb.latitudeOfLastGridPointInDegrees)



def init_grid(debug=None):
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

    grid_lons = grid[0]
    grid_lats = grid[1]

    lats = np.array([point1[0], point2[0]])
    lons = np.array([point1[1], point2[1]])

    min_lon = np.where(grid_lons == np.amin(lons))
    max_lon = np.where(grid_lons == np.amax(lons))

    min_lat = np.where(grid_lats == np.amin(lats))
    max_lat = np.where(grid_lats == np.amax(lats))

    indices = {'min_lon': min_lon[0][0], 'max_lon': max_lon[0][0], 'min_lat': min_lat[0][0], 'max_lat': max_lat[0][0]}

    if (debug):
        print('min lon idx:', indices['min_lon'])
        print('max lon idx:', indices['max_lon'])
        print('min lat idx:', indices['min_lat'])
        print('max lat idx:', indices['max_lat'])
        print('------------------------------------')

    return indices



def trunc(vals, decs=0):
    return np.trunc(vals*10**decs)/(10**decs)



def subset_grid(grid, bbox, debug=False):
    x_min = bbox['min_lon']
    x_max = bbox['max_lon']
    y_min = bbox['min_lat']
    y_max = bbox['max_lat']

    lons = grid[0][x_min : x_max + 1]
    lats = grid[1][y_max : y_min]

    if (debug):
        print('lons (x) length:', len(lons))
        print('lats (y) length:', len(lats))
        print('------------------------------------')

    return (lons, lats)



def subset_data(bbox, data, debug=False):

    x_min = bbox['min_lon']
    x_max = bbox['max_lon']
    y_min = bbox['min_lat']
    y_max = bbox['max_lat']

    subset = data[y_max : y_min, x_min : x_max + 1]

    subset[subset <= 0] = float('nan')

    if (debug):
        print('min x idx:', x_min)
        print('max x idx:', x_max)
        print('min y idx:', y_min)
        print('max y idx:', y_max)

        print('subset shape (y, x):', subset.shape)
        print('------------------------------------')

    return subset




def main():

    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms'
    #f_name = 'MRMS_MergedReflectivityQC_00.50_20190523-212434.grib2'

    f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905/MergedReflectivityQC_01.50'
    f_name = 'MRMS_MergedReflectivityQC_01.50_20190523-212434.grib2'

    f_abs = join(f_path, f_name)

    grid = init_grid()
    
    grb_obj = get_grb_data(f_abs)

    point1 = (37.195, -102.185)
    point2 = (34.565, -99.865)

    bbox = get_bbox_indices(grid, point1, point2)
    data_subs = subset_data(bbox, grb_obj.data)

    grb_obj.set_data(data_subs)

    grid_subs = subset_grid(grid, bbox)

    grb_obj.set_grid_lons(grid_subs[0])
    grb_obj.set_grid_lats(grid_subs[1])

    plot_grb_data(grb_obj)




if (__name__ == '__main__'):
    main()