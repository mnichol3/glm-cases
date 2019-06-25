import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.ndimage
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.cm as cm

from glm_utils import georeference


def plot_mercator_dual(data_dict, extent_coords, wtlma_obj):


    globe = ccrs.Globe(semimajor_axis=data_dict['semi_major_axis'], semiminor_axis=data_dict['semi_minor_axis'],
                       flattening=None, inverse_flattening=data_dict['inv_flattening'])

    ext_lats = [extent_coords[0][0], extent_coords[1][0]]
    ext_lons = [extent_coords[0][1], extent_coords[1][1]]

    Xs, Ys = georeference(data_dict['x'], data_dict['y'], data_dict['lon_0'], data_dict['height'],
                          data_dict['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    ax.set_extent([min(ext_lons), max(ext_lons), min(ext_lats), max(ext_lats)], crs=ccrs.PlateCarree())

    cmesh = plt.pcolormesh(Xs, Ys, data_dict['data'], vmin=0, vmax=350, transform=ccrs.PlateCarree(), cmap=cm.jet)
    cbar = plt.colorbar(cmesh,fraction=0.046, pad=0.04)

    scat = plt.scatter(wtlma_obj.data['lon'], wtlma_obj.data['lat'], c=wtlma_obj.data['P'], marker="2", cmap=cm.gist_ncar_r, vmin=-20, vmax=100, transform=ccrs.PlateCarree())

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
