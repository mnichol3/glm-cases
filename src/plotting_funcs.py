import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.ndimage
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.cm as cm
import matplotlib.colors as colors

from glm_utils import georeference


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
                       marker="2", s=150, cmap=cm.gist_ncar_r, vmin=-20, vmax=100, zorder=3, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(scat, fraction=0.046, pad=0.04)
    cbar2.set_label('WTLMA Flash Power (dBW)')

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
