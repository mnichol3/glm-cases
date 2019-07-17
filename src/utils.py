import numpy as np
from pyproj import Geod
import pyproj
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.colors as colors
from os.path import join

import plotting_utils
import glm_utils

def test_glm_plot(abs_path):
    glm_obj = glm_utils.read_file(abs_path, window=True, meta=True)

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    Xs, Ys = glm_utils.georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp', zorder=0)

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    ax.set_extent([-102.5, -100, 35, 36.5], crs=ccrs.PlateCarree())

    lon_ticks = [x for x in np.arange(-180, 181, 0.5)]
    lat_ticks = [x for x in np.arange(-90, 91, 0.5)]

    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = plt.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=ccrs.PlateCarree(), cmap=cm.jet, zorder=2)

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional', fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.set_label('GLM Flash Extent Density')

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


    plt.title('GLM FED {} {}'.format(glm_obj.scan_date, glm_obj.scan_time, loc='right'))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



f1 = 'IXTR99_KNES_232107_40255.2019052321'
f2 = 'IXTR99_KNES_232107_14608.2019052322'
base_path = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523'
abs_path = join(base_path, f2)
test_glm_plot(abs_path)
