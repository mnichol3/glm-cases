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
import sys

import plotting_utils
import glm_utils

def test_glm_plot(abs_path):
    glm_obj = glm_utils.read_file(abs_path, window=False, meta=True)

    tx_county_path = '/home/mnichol3/Coding/glm-cases/resources/Texas_County_Boundaries/Texas_County_Boundaries.shp'
    ok_county_path = '/home/mnichol3/Coding/glm-cases/resources/tl_2016_40_cousub/tl_2016_40_cousub.shp'

    tx_counties_reader = shpreader.Reader(tx_county_path)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(ok_county_path)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    Xs, Ys = glm_utils.georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp', zorder=0)

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=1)
    ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=1)

    #ax.set_extent([-102.5, -100, 35, 36.5], crs=ccrs.PlateCarree())
    ax.set_extent([-103, -100.25, 35.5, 37], crs=ccrs.PlateCarree())

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



def are_equal(file1, file2, base_path):
    path1 = join(base_path, file1)
    path2 = join(base_path, file2)

    glm_obj_1 = glm_utils.read_file(path1)
    glm_obj_2 = glm_utils.read_file(path2)

    print(np.array_equal(glm_obj_1.data, glm_obj_2.data))



def dump_shp(abs_path, pretty=False):
    count = 0
    reader = shpreader.Reader(abs_path)
    if (pretty):
        for rec in reader.records():
            count += 1
            for key, val in rec.attributes.items():
                print('{}: {}'.format(key, val))
            print('-'*25)
    else:
        for rec in reader.records():
            count += 1
            print(rec.attributes)
            print('-'*25)
    print('\nTotal records: {}\n'.format(count))



def plot_wwa(abs_path, datetime):
    """
    To plot warning polygons and not county/zone/parish, must filter wwa records by
    GTYPE = P
    """
    tx_county_path = '/home/mnichol3/Coding/glm-cases/resources/Texas_County_Boundaries/Texas_County_Boundaries.shp'
    ok_county_path = '//home/mnichol3/Coding/glm-cases/resources/tl_2016_40_cousub/tl_2016_40_cousub.shp'

    wwa_reader = shpreader.Reader(abs_path)

    # Datetime format: 201905232344
    filtered_wwa_sv = [rec.geometry for rec in wwa_reader.records() if (rec.attributes['GTYPE'] == 'P')
                    and (_valid_wwa_time(rec.attributes['ISSUED'], rec.attributes['EXPIRED'], datetime))
                    and (rec.attributes['PHENOM'] == 'SV')]
    filtered_wwa_to = [rec.geometry for rec in wwa_reader.records() if (rec.attributes['GTYPE'] == 'P')
                    and (_valid_wwa_time(rec.attributes['ISSUED'], rec.attributes['EXPIRED'], datetime))
                    and (rec.attributes['PHENOM'] == 'TO')]

    sv_polys = cfeature.ShapelyFeature(filtered_wwa_sv, ccrs.PlateCarree())
    to_polys = cfeature.ShapelyFeature(filtered_wwa_to, ccrs.PlateCarree())

    tx_counties_reader = shpreader.Reader(tx_county_path)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(ok_county_path)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator())

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp', zorder=0)

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=1)
    ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=1)
    ax.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=1)
    ax.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=1)

    #ax.set_extent([-102.5, -100, 35, 36.5], crs=ccrs.PlateCarree())
    ax.set_extent([-103, -100.25, 35.5, 37], crs=ccrs.PlateCarree())

    lon_ticks = [x for x in np.arange(-180, 181, 0.5)]
    lat_ticks = [x for x in np.arange(-90, 91, 0.5)]

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

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def calc_geo_area():
    import pyproj
    import shapely
    import shapely.ops as ops
    from shapely.geometry.polygon import Polygon
    from functools import partial

             # x0       x1       y0      y1
    extent = [-102.443, -100.00, 35.362, 36.992]

    poly = Polygon([(extent[0], extent[2]), (extent[0], extent[3]),
                    (extent[1], extent[3]), (extent[1], extent[2]),
                    (extent[0], extent[2])])

    print('Polygon bounds: {}'.format(poly.bounds))

    poly_aea = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=poly.bounds[1],
                lat2=poly.bounds[3]
            )
        ),
        poly
    )

    print('Area (km^2): {}'.format(poly_aea.area/1000))


def _valid_wwa_time(issued, expired, target):
    target = int(target)
    expired = int(expired)
    issued = int(issued)
    return (target >= issued and target <= expired)



def _dump_garbage():
    import gc

    print('\nGARBAGE')
    gc.collect()
    print('\nGARBAGE OBJECTS')
    for x in gc.garbage:
        s = str(x)
        if (len(s > 80)):
            s = s[:80]
        print('{} \n {}'.format(type(x), s))

# f1 = 'IXTR99_KNES_232107_40255.2019052321'
# f2 = 'IXTR99_KNES_232107_14608.2019052322'
# base_path = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523'
# wwa_base = '/home/mnichol3/Coding/glm-cases/resources/wwa_201905230000_201905240000'
# wwa_fname = 'wwa_201905230000_201905240000.shp'
# wwa_abs_path = join(wwa_base, wwa_fname)
# #plot_wwa(wwa_abs_path, '201905232120')
# dump_shp(wwa_abs_path, pretty=True)

# path = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523'
# f1 = 'IXTR99_KNES_232059_14586.2019052322.nc'
# f2 = 'IXTR99_KNES_232100_14590.2019052322.nc'
# # f3 = 'IXTR99_KNES_232059_14586.2019052321.nc'
# f4 = 'IXTR99_KNES_232100_40226.2019052321.nc'
# # test_glm_plot(join(path, f3))
# test_glm_plot(join(path, f4))
