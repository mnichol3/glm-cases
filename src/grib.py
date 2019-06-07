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


def print_keys(fname):
    grb = pygrib.open(fname)
    grb = grb[1]

    for key in grb.keys():
        print(key)



def get_keys(fname):
    grb = pygrib.open(fname)
    grb = grb[1]

    return grb.keys()



def plot_grb(fname):

    grb = pygrib.open(fname)
    grb = grb[1]

    major_ax = grb.earthMajorAxis
    minor_ax = grb.earthMinorAxis

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



def get_files_in_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    return files



def main():

    f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms'
    f_name = 'MRMS_MergedReflectivityQC_01.00_20190523-212434.grib2'

    f_abs = join(f_path, f_name)

    #plot_grb(fname)
    print(get_files_in_dir(f_path))



if (__name__ == '__main__'):
    main()
