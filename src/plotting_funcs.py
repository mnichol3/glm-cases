import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.ndimage
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.ndimage
import re
from matplotlib.patches import Polygon
from os.path import join

from glm_utils import georeference
import grib
from plotting_utils import to_file, load_data, load_coordinates, parse_coord_fnames, process_slice, process_slice_inset
from plotting_utils import geodesic_point_buffer
from proj_utils import scan_to_geod


TX_SHP_PATH = '/home/mnichol3/Coding/glm-cases/resources/Texas_County_Boundaries/Texas_County_Boundaries.shp'
OK_SHP_PATH = '/home/mnichol3/Coding/glm-cases/resources/tl_2016_40_cousub/tl_2016_40_cousub.shp'


def plot_mercator_dual(glm_obj, wtlma_obj, grid_extent=None, points_to_plot=None,
                       range_rings=False, wwa_polys=None, satellite_data=None,
                       show=True, save=False, outpath=None):
    """
    Plots both GLM FED, as a colormesh, and WTLMA sources, as points.

    Parameters
    ----------
    glm_obj : LocalGLMFile
    wtlma_obj : LocalWTLMAFile
    grid_extent : dictionary
        Dictionary that defines the extent of the data grid
        Keys: min_lon, max_lon, min_lat, max_lat
    points_to_plot : tuple of tuples or list of tuples, optional
        Format: [(lat1, lon1), (lat2, lon2)]
    range_rings : bool, optional
        If true, plots color-coded WTLMA range-rings to indicate the possibly
        decrease in data quality due to distance
    wwa_polys : dict; key : str, value : polygon; optional
        NWS Severe Thunderstorm and/or Tornado warning polygons
    satellite_data: dict, optional
        Dictionary of satellite data & metadata to plot
    show : bool, optional
        If True, the plot will be displayed in the matplotlib GUI. Default is True
    save : bool, optional
        If True, the plot will be saved to the location specified by the outpath
        parameter, which must be specified. Default is False
    outpath : str, optional
        The path to the directory to save the plot. If save is True, outpath
        cannot be None
    """
    z_ord = {'map': 9, 'sat_vis': 2, 'sat_inf': 3, 'glm': 4, 'lma': 5, 'wwa': 6, 'top': 10}

    crs_plt = ccrs.PlateCarree() # Globe keyword was messing everything up

    tx_counties_reader = shpreader.Reader(TX_SHP_PATH)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, crs_plt)

    ok_counties_reader = shpreader.Reader(OK_SHP_PATH)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, crs_plt)

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111, projection=ccrs.Mercator())

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, facecolor='none', linewidth=.8, edgecolor='gray', zorder=z_ord['map'])
    ax.add_feature(tx_counties, linewidth=.2, facecolor='none', edgecolor='gray', zorder=z_ord['map'])
    ax.add_feature(ok_counties, linewidth=.2, facecolor='none', edgecolor='gray', zorder=z_ord['map'])

    cent_lat = float(wtlma_obj.coord_center[0])
    cent_lon = float(wtlma_obj.coord_center[1])

    if (grid_extent is None):
        bounds = geodesic_point_buffer(cent_lat, cent_lon, 300)
        lats = [float(x[1]) for x in bounds.coords[:]]
        lons = [float(x[0]) for x in bounds.coords[:]]
        extent = {'min_lon': min(lons), 'max_lon': max(lons), 'min_lat': min(lats), 'max_lat': max(lats)}
        del lats
        del lons
    else:
        extent = grid_extent

    ax.set_extent([extent['min_lon'], extent['max_lon'], extent['min_lat'], extent['max_lat']], crs=crs_plt)

    grid_lons = np.arange(extent['min_lon'], extent['max_lon'], 0.01)
    grid_lats = np.arange(extent['min_lat'], extent['max_lat'], 0.01)

    ############################## Plot GLM data ##############################
    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = plt.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=crs_plt,
                        cmap=cm.jet, zorder=z_ord['glm'])

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional',
                        fraction=0.046, pad=0.08, shrink=0.85)
    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label('GLM Flash Extent Density', fontsize=8)

    ############################## Plot LMA data ##############################
    scat = plt.scatter(wtlma_obj.data['lon'], wtlma_obj.data['lat'], c=wtlma_obj.data['P'],
                       marker='o', s=20, cmap=cm.gist_ncar_r, vmin=-20, vmax=100,
                       zorder=z_ord['lma'], transform=crs_plt)
    cbar2 = plt.colorbar(scat, fraction=0.046, pad=0.08, shrink=0.85)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('WTLMA Source Power (dBW)', fontsize=8)

    ############################ Plot x-sect coords ############################
    if (points_to_plot is not None):
        plt.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='r', zorder=z_ord['top'], transform=crs_plt)

    if (range_rings):
        clrs = ['g', 'y']
        for idx, x in enumerate([100, 250]):
            coord_list = geodesic_point_buffer(cent_lat, cent_lon, x)
            lats = [float(x[1]) for x in coord_list.coords[:]]
            max_lat = max(lats)

            # https://stackoverflow.com/questions/27574897/plotting-disconnected-entities-with-shapely-descartes-and-matplotlib
            mpl_poly = Polygon(np.array(coord_list), ec=clrs[idx], fc="none", transform=crs_plt,
                               linewidth=1.25, zorder=z_ord['map'])
            ax.add_patch(mpl_poly)

    if (satellite_data):
        if (len(satellite_data) != 2):
            print('Error: Invalid satellite data params to produce sandwhich image')
        else:
            visual = satellite_data[0]
            infrared = satellite_data[1]

            sat_height = visual['sat_height']
            sat_lon = visual['sat_lon']
            sat_sweep = visual['sat_sweep']
            scan_date = visual['scan_date']

            y_min, x_min = scan_to_geod(min(visual['y_image_bounds']), min(visual['x_image_bounds']))
            y_max, x_max = scan_to_geod(max(visual['y_image_bounds']), max(visual['x_image_bounds']))

            crs_geos = ccrs.Geostationary(central_longitude=sat_lon, satellite_height=sat_height,
                                           false_easting=0, false_northing=0, globe=globe, sweep_axis=sat_sweep)

            trans_pts = crs_geos.transform_points(crs_plt, np.array([x_min, x_max]), np.array([y_min, y_max]))

            proj_extent = (min(trans_pts[0][0], trans_pts[1][0]), max(trans_pts[0][0], trans_pts[1][0]),
                           min(trans_pts[0][1], trans_pts[1][1]), max(trans_pts[0][1], trans_pts[1][1]))

            viz_img = plt.imshow(visual['data'], cmap=cm.Greys_r, extent=proj_extent, origin='upper',
                                 vmin=visual['min_data_val'], vmax=visual['max_data_val'],
                                 zorder=z_ord['sat_vis'], transform=crs_geos)

            # infrared_norm = colors.LogNorm(vmin=190, vmax=270)
            # inf_img = plt.imshow(infrared['data'], cmap=cm.nipy_spectral_r, origin='upper',
            #                      norm=infrared_norm, extent=proj_extent, zorder=z_ord['sat_inf'],
            #                      alpha=0.4, transform=crs_geos)
            #
            # cbar_bounds = np.arange(190, 270, 10)
            # cbar_sat = plt.colorbar(inf_img, ticks=cbar_bounds, spacing='proportional', fraction=0.046, pad=0.04)
            # cbar_sat.ax.set_yticklabels([str(x) for x in cbar_bounds])
            # cbar_sat.set_label('Cloud-top Temperature (K)')

            inf_img = plt.imshow(infrared['data'], cmap=cm.nipy_spectral_r, origin='upper',
                                 vmin=190, vmax=270, extent=proj_extent, zorder=z_ord['sat_inf'],
                                 alpha=0.4, transform=crs_geos)

            cbar_bounds = np.arange(190, 270, 10)
            cbar_sat = plt.colorbar(inf_img, ticks=[x for x in cbar_bounds], spacing='proportional',
                            fraction=0.046, pad=0.08, shrink=0.85)
            cbar_sat.set_ticklabels([str(x) for x in cbar_bounds], update_ticks=True)
            cbar_sat.ax.tick_params(labelsize=6)
            cbar_sat.set_label('Cloud-top Temperature (K)', fontsize=8)

    if (wwa_polys is not None):
        wwa_keys = wwa_polys.keys()

        if ('SV' in wwa_keys):
            sv_polys = cfeature.ShapelyFeature(wwa_polys['SV'], crs_plt)
            ax.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=z_ord['wwa'])
        if ('TO' in wwa_keys):
            to_polys = cfeature.ShapelyFeature(wwa_polys['TO'], crs_plt)
            ax.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=z_ord['wwa'])

    plt.title('GLM Flash Extent Density {} {}z\n WTLMA Sources {}z'.format(
            glm_obj.scan_date, glm_obj.scan_time, wtlma_obj._start_time_pp()), loc='right',
            fontsize=8)

    plt.gca().set_aspect('equal', adjustable='box')
    if (save):
        if (outpath is not None):
            fname = 'plan-{}-{}z.png'.format(glm_obj.scan_date, glm_obj.scan_time)
            path = join(outpath, fname)
            plt.savefig(path, dpi=500)
        else:
            raise ValueError('Error: Outpath cannot be None')
    if (show):
        plt.show()
    plt.close('all')



def plot_mercator_dual_2(glm_obj, wtlma_obj, grid_extent=None, points_to_plot=None,
                         range_rings=False, wwa_polys=None, satellite_data=None,
                         show=True, save=False, outpath=None):
    """
    Same as plot_mercator_dual(), except it plots the wtlma strokes as
    power densities

    Parameters
    ----------
    glm_obj : LocalGLMFile
    wtlma_obj : LocalWTLMAFile
    grid_extent : dictionary, optional
        Dictionary that defines the extent of the data grid
        Keys: min_lon, max_lon, min_lat, max_lat
        If none is given, the plot is centered on the WTLMA centroid
    points_to_plot : tuple of tuples or list of tuples, optional
        Format: [(lat1, lon1), (lat2, lon2)]
    range_rings : bool, optional
        If true, plots color-coded WTLMA range-rings to indicate the possibly
        decrease in data quality due to distance
    wwa_polys : dict; key : str, value : polygon; optional
        NWS Severe Thunderstorm and/or Tornado warning polygons
    satellite_data: dict, optional
        Dictionary of satellite data & metadata to plot
    show : bool, optional
        If True, the plot will be displayed in the matplotlib GUI. Default is True
    save : bool, optional
        If True, the plot will be saved to the location specified by the outpath
        parameter, which must be specified. Default is False
    outpath : str, optional
        The path to the directory to save the plot. If save is True, outpath
        cannot be None
    """
    z_ord = {'map':6 , 'sat_vis': 1, 'sat_inf': 2, 'glm': 3, 'lma': 4, 'wwa': 5, 'top': 10}
    tx_counties_reader = shpreader.Reader(TX_SHP_PATH)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(OK_SHP_PATH)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    crs_plt = ccrs.PlateCarree() # Globe keyword was messing everything up

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111, projection=ccrs.Mercator())

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp', zorder=0)

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])
    ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])

    cent_lat = float(wtlma_obj.coord_center[0])
    cent_lon = float(wtlma_obj.coord_center[1])

    if (grid_extent is None):
        bounds = geodesic_point_buffer(cent_lat, cent_lon, 300)
        lats = [float(x[1]) for x in bounds.coords[:]]
        lons = [float(x[0]) for x in bounds.coords[:]]
        extent = {'min_lon': min(lons), 'max_lon': max(lons), 'min_lat': min(lats), 'max_lat': max(lats)}
        del lats
        del lons
    else:
        extent = grid_extent

    ax.set_extent([extent['min_lon'], extent['max_lon'], extent['min_lat'], extent['max_lat']], crs=crs_plt)

    grid_lons = np.arange(extent['min_lon'], extent['max_lon'], 0.01)
    grid_lats = np.arange(extent['min_lat'], extent['max_lat'], 0.01)

    ############################## Plot GLM data ##############################
    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = plt.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=crs_plt,
                        cmap=cm.jet, zorder=z_ord['glm'])

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional',
                        fraction=0.046, pad=0.08, shrink=0.85)
    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label('GLM Flash Extent Density', fontsize=8)

    ############################## Plot LMA data ##############################
    lma_norm = colors.LogNorm(vmin=1, vmax=400)

    H, X_edges, Y_edges = np.histogram2d(wtlma_obj.data['lon'], wtlma_obj.data['lat'],
                          bins=100, range=[[extent['min_lon'], extent['max_lon']], [extent['min_lat'], extent['max_lat']]],
                          weights=wtlma_obj.data['P']) # bins=[len(grid_lons), len(grid_lats)]

    lma_mesh = plt.pcolormesh(X_edges, Y_edges, H.T, norm=lma_norm, transform=crs_plt,
                        cmap=cm.inferno, zorder=z_ord['lma'])

    lma_bounds = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400]
    cbar2 = plt.colorbar(lma_mesh, ticks=lma_bounds, spacing='proportional',fraction=0.046,
                        pad=0.08, shrink=0.85)
    cbar2.ax.set_yticklabels([str(x) for x in lma_bounds])
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('WTLMA Source Power Density (dBW)', fontsize=8)

    ############################ Plot x-sect coords ############################
    if (points_to_plot is not None):
        plt.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='r', zorder=z_ord['wwa'], transform=crs_plt)

    if (range_rings):
        clrs = ['g', 'y']
        for idx, x in enumerate([100, 250]):
            coord_list = geodesic_point_buffer(cent_lat, cent_lon, x)
            lats = [float(x[1]) for x in coord_list.coords[:]]
            max_lat = max(lats)

            # https://stackoverflow.com/questions/27574897/plotting-disconnected-entities-with-shapely-descartes-and-matplotlib
            mpl_poly = Polygon(np.array(coord_list), ec=clrs[idx], fc="none", transform=crs_plt,
                               linewidth=1.25, zorder=z_ord['map'])
            ax.add_patch(mpl_poly)

    if (satellite_data):
        if (len(satellite_data) != 2):
            print('Error: Invalid satellite data params to produce sandwhich image')
        else:
            visual = satellite_data[0]
            infrared = satellite_data[1]

            sat_height = visual['sat_height']
            sat_lon = visual['sat_lon']
            sat_sweep = visual['sat_sweep']
            scan_date = visual['scan_date']

            y_min, x_min = scan_to_geod(min(visual['y_image_bounds']), min(visual['x_image_bounds']))
            y_max, x_max = scan_to_geod(max(visual['y_image_bounds']), max(visual['x_image_bounds']))

            crs_geos = ccrs.Geostationary(central_longitude=sat_lon, satellite_height=sat_height,
                                           false_easting=0, false_northing=0, globe=globe, sweep_axis=sat_sweep)

            trans_pts = crs_geos.transform_points(crs_plt, np.array([x_min, x_max]), np.array([y_min, y_max]))

            proj_extent = (min(trans_pts[0][0], trans_pts[1][0]), max(trans_pts[0][0], trans_pts[1][0]),
                           min(trans_pts[0][1], trans_pts[1][1]), max(trans_pts[0][1], trans_pts[1][1]))

            viz_img = plt.imshow(visual['data'], cmap=cm.Greys_r, extent=proj_extent, origin='upper',
                                 vmin=visual['min_data_val'], vmax=visual['max_data_val'],
                                 zorder=z_ord['sat_vis'], transform=crs_geos)

            inf_img = plt.imshow(infrared['data'], cmap=cm.nipy_spectral_r, origin='upper',
                                 vmin=190, vmax=270, extent=proj_extent, zorder=z_ord['sat_inf'],
                                 alpha=0.4, transform=crs_geos)


            cbar_bounds = np.arange(190, 270, 10)
            cbar_sat = plt.colorbar(inf_img, ticks=[x for x in cbar_bounds], spacing='proportional',
                                fraction=0.046, pad=0.08, shrink=0.85)
            cbar_sat.set_ticklabels([str(x) for x in cbar_bounds], update_ticks=True)
            cbar_sat.ax.tick_params(labelsize=6)
            cbar_sat.set_label('Cloud-top Temperature (K)', fontsize=8)

    if (wwa_polys is not None):
        wwa_keys = wwa_polys.keys()

        if ('SV' in wwa_keys):
            sv_polys = cfeature.ShapelyFeature(wwa_polys['SV'], ccrs.PlateCarree())
            ax.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=z_ord['wwa'])
        if ('TO' in wwa_keys):
            to_polys = cfeature.ShapelyFeature(wwa_polys['TO'], ccrs.PlateCarree())
            ax.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=z_ord['wwa'])

    plt.title('GLM Flash Extent Density {} {}z\n WTLMA Power-Weighted Source Density {}z'.format(
                glm_obj.scan_date, glm_obj.scan_time, wtlma_obj._start_time_pp()), loc='right',
                fontsize=8)
    # plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    if (save):
        if (outpath is not None):
            fname = 'plan-{}-{}z.png'.format(glm_obj.scan_date, glm_obj.scan_time)
            path = join(outpath, fname)
            plt.savefig(path, dpi=500, bbox_inches='tight')
        else:
            raise ValueError('Error: Outpath cannot be None')
    if (show):
        plt.show()
    plt.close('all')



def plot_merc_glm_lma_sbs(glm_obj, wtlma_obj, grid_extent=None, points_to_plot=None,
                         range_rings=False, wwa_polys=None, satellite_data=None,
                         show=True, save=False, outpath=None):
    """
    Creates a figure with two side-by-side subplots. The first subplot displays
    the GLM Flash Extent Density as a color mesh. The second subplot displays
    the WTLMA power-weighted source density. Both subplots are plotted over
    a map & GOES-16 ABI Ch.2 & Ch.13 'Sandwhich' imagery

    Parameters
    ----------
    glm_obj : LocalGLMFile
    wtlma_obj : LocalWTLMAFile
    grid_extent : dictionary
        Dictionary that defines the extent of the data grid
        Keys: min_lon, max_lon, min_lat, max_lat
    points_to_plot : tuple of tuples or list of tuples, optional
        Format: [(lat1, lon1), (lat2, lon2)]
    range_rings : bool, optional
        If true, plots color-coded WTLMA range-rings to indicate the possibly
        decrease in data quality due to distance
    wwa_polys : dict; key : str, value : polygon; optional
        NWS Severe Thunderstorm and/or Tornado warning polygons
    satellite_data: dict, optional
        Dictionary of satellite data & metadata to plot
    show : bool, optional
        If True, the plot will be displayed in the matplotlib GUI. Default is True
    save : bool, optional
        If True, the plot will be saved to the location specified by the outpath
        parameter, which must be specified. Default is False
    outpath : str, optional
        The path to the directory to save the plot. If save is True, outpath
        cannot be None
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    z_ord = {'map':6 , 'sat_vis': 1, 'sat_inf': 2, 'glm': 3, 'lma': 4, 'wwa': 5, 'top': 10}
    tx_counties_reader = shpreader.Reader(TX_SHP_PATH)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(OK_SHP_PATH)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    cent_lat = float(wtlma_obj.coord_center[0])
    cent_lon = float(wtlma_obj.coord_center[1])

    if (grid_extent is None):
        bounds = geodesic_point_buffer(cent_lat, cent_lon, 300)
        lats = [float(x[1]) for x in bounds.coords[:]]
        lons = [float(x[0]) for x in bounds.coords[:]]
        extent = {'min_lon': min(lons), 'max_lon': max(lons), 'min_lat': min(lats), 'max_lat': max(lats)}
        del lats
        del lons
    else:
        extent = grid_extent

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    crs_plt = ccrs.PlateCarree() # Globe keyword was messing everything up

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Mercator()},
                        figsize=(12, 8))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                        name='admin_1_states_provinces_shp', zorder=0)

    for ax in [ax1, ax2]:
        ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)
        ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])
        ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])
        ax.set_extent([extent['min_lon'], extent['max_lon'], extent['min_lat'], extent['max_lat']], crs=crs_plt)

    grid_lons = np.arange(extent['min_lon'], extent['max_lon'], 0.01)
    grid_lats = np.arange(extent['min_lat'], extent['max_lat'], 0.01)

    ############################## Plot GLM data ##############################
    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = ax1.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=crs_plt,
                        cmap=cm.jet, zorder=z_ord['glm'])

    axins1 = inset_axes(ax1, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0., 1, 1),
                        bbox_transform=ax1.transAxes, borderpad=0)

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional',
                        fraction=0.046, pad=0, cax=axins1)

    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label('GLM Flash Extent Density', fontsize=8)

    ############################## Plot LMA data ##############################
    lma_norm = colors.LogNorm(vmin=1, vmax=400)

    H, X_edges, Y_edges = np.histogram2d(wtlma_obj.data['lon'], wtlma_obj.data['lat'],
                          bins=100, range=[[extent['min_lon'], extent['max_lon']], [extent['min_lat'], extent['max_lat']]],
                          weights=wtlma_obj.data['P']) # bins=[len(grid_lons), len(grid_lats)]

    lma_mesh = ax2.pcolormesh(X_edges, Y_edges, H.T, norm=lma_norm, transform=crs_plt,
                        cmap=cm.inferno, zorder=z_ord['lma'])

    lma_bounds = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400]

    axins2 = inset_axes(ax2, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0., 1, 1),
                    bbox_transform=ax2.transAxes, borderpad=0)

    cbar2 = plt.colorbar(lma_mesh, ticks=lma_bounds, spacing='proportional',fraction=0.046,
                    pad=0.02, cax=axins2)

    cbar2.ax.set_yticklabels([str(x) for x in lma_bounds])
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('WTLMA Source Power Density (dBW)', fontsize=8)

    ############################ Plot x-sect coords ############################
    if (points_to_plot is not None):
        ax1.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='r', zorder=z_ord['wwa'], transform=crs_plt)
        ax2.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='r', zorder=z_ord['wwa'], transform=crs_plt)

    if (range_rings):
        clrs = ['g', 'y']
        for idx, x in enumerate([100, 250]):
            coord_list = geodesic_point_buffer(cent_lat, cent_lon, x)
            lats = [float(x[1]) for x in coord_list.coords[:]]
            max_lat = max(lats)

            # Only way mpl won't throw a fit about using the patch twice
            for ax in [ax1, ax2]:
                # https://stackoverflow.com/questions/27574897/plotting-disconnected-entities-
                # with-shapely-descartes-and-matplotlib
                mpl_poly = Polygon(np.array(coord_list), ec=clrs[idx], fc="none", transform=crs_plt,
                                   linewidth=1.25, zorder=z_ord['map'])
                ax.add_patch(mpl_poly)

    if (satellite_data):
        if (len(satellite_data) != 2):
            print('Error: Invalid satellite data params to produce sandwhich image')
        else:
            visual = satellite_data[0]
            infrared = satellite_data[1]

            sat_height = visual['sat_height']
            sat_lon = visual['sat_lon']
            sat_sweep = visual['sat_sweep']
            scan_date = visual['scan_date']

            y_min, x_min = scan_to_geod(min(visual['y_image_bounds']), min(visual['x_image_bounds']))
            y_max, x_max = scan_to_geod(max(visual['y_image_bounds']), max(visual['x_image_bounds']))

            crs_geos = ccrs.Geostationary(central_longitude=sat_lon, satellite_height=sat_height,
                                           false_easting=0, false_northing=0, globe=globe, sweep_axis=sat_sweep)

            trans_pts = crs_geos.transform_points(crs_plt, np.array([x_min, x_max]), np.array([y_min, y_max]))

            proj_extent = (min(trans_pts[0][0], trans_pts[1][0]), max(trans_pts[0][0], trans_pts[1][0]),
                           min(trans_pts[0][1], trans_pts[1][1]), max(trans_pts[0][1], trans_pts[1][1]))

            viz_img1 = ax1.imshow(visual['data'], cmap=cm.Greys_r, extent=proj_extent, origin='upper',
                                 vmin=visual['min_data_val'], vmax=visual['max_data_val'],
                                 zorder=z_ord['sat_vis'], transform=crs_geos)

            inf_img1 = ax1.imshow(infrared['data'], cmap=cm.nipy_spectral_r, origin='upper',
                                 vmin=190, vmax=270, extent=proj_extent, zorder=z_ord['sat_inf'],
                                 alpha=0.4, transform=crs_geos)

            viz_img2 = ax2.imshow(visual['data'], cmap=cm.Greys_r, extent=proj_extent, origin='upper',
                                 vmin=visual['min_data_val'], vmax=visual['max_data_val'],
                                 zorder=z_ord['sat_vis'], transform=crs_geos)

            inf_img2 = ax2.imshow(infrared['data'], cmap=cm.nipy_spectral_r, origin='upper',
                                 vmin=190, vmax=270, extent=proj_extent, zorder=z_ord['sat_inf'],
                                 alpha=0.4, transform=crs_geos, aspect="auto")


            cbar_bounds = np.arange(190, 270, 10)
            cbar_sat = fig.colorbar(inf_img1, ticks=[x for x in cbar_bounds], spacing='proportional',
                                fraction=0.046, pad=0.02, shrink=0.53, ax=[ax1, ax2],
                                orientation='horizontal')
            cbar_sat.set_ticklabels([str(x) for x in cbar_bounds], update_ticks=True)
            cbar_sat.ax.tick_params(labelsize=6)
            cbar_sat.set_label('Cloud-top Temperature (K)', fontsize=8)

    if (wwa_polys is not None):
        wwa_keys = wwa_polys.keys()

        if ('SV' in wwa_keys):
            sv_polys = cfeature.ShapelyFeature(wwa_polys['SV'], ccrs.PlateCarree())
            ax1.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=z_ord['wwa'])
            ax2.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=z_ord['wwa'])
        if ('TO' in wwa_keys):
            to_polys = cfeature.ShapelyFeature(wwa_polys['TO'], ccrs.PlateCarree())
            ax1.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=z_ord['wwa'])
            ax2.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=z_ord['wwa'])

    ax1.set_title('GLM Flash Extent Density {} {}z'.format(glm_obj.scan_date, glm_obj.scan_time),
                   loc='center', fontsize=8)
    ax2.set_title('WTLMA Power-Weighted Source Density {}z'.format(wtlma_obj._start_time_pp()),
                   loc='center', fontsize=8)

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    if (save):
        if (outpath is not None):
            fname = 'sbs-plan-{}-{}z.png'.format(glm_obj.scan_date, glm_obj.scan_time)
            path = join(outpath, fname)
            plt.savefig(path, dpi=500, bbox_inches='tight')
        else:
            raise ValueError('Error: Outpath cannot be None')
    if (show):
        plt.show()
    plt.close('all')



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



def plot_merc_abi_mrms(sat_data, mrms_obj, grid_extent=None, points_to_plot=None,
                       range_rings=False, wwa_polys=None, show=True, save=False,
                       outpath=None):
    """
    sat-data only consists on inf data

    points_to_plot : list of tuples, optional
        Coordinate pairs to plot
        Format: [(lat1, lon1), (lat2, lon2)]
    """

    z_ord = {'map':6 , 'mrms': 1, 'sat': 2, 'glm': 3, 'lma': 4, 'wwa': 5, 'top': 10}

    sat_height = sat_data['sat_height']
    sat_lon = sat_data['sat_lon']
    sat_sweep = sat_data['sat_sweep']
    scan_date = sat_data['scan_date']

    tx_counties_reader = shpreader.Reader(TX_SHP_PATH)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(OK_SHP_PATH)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    if (grid_extent is None):
        extent = [min(mrms_obj.grid_lons), max(mrms_obj.grid_lons),
                  min(mrms_obj.grid_lats), max(mrms_obj.grid_lats)]
    else:
        extent = grid_extent

    y_min, x_min = scan_to_geod(min(sat_data['y_image_bounds']), min(sat_data['x_image_bounds']))
    y_max, x_max = scan_to_geod(max(sat_data['y_image_bounds']), max(sat_data['x_image_bounds']))

    globe = ccrs.Globe(semimajor_axis=sat_data['semimajor_ax'], semiminor_axis=sat_data['semiminor_ax'],
                       flattening=None, inverse_flattening=sat_data['inverse_flattening'])

    crs_geos = ccrs.Geostationary(central_longitude=sat_lon, satellite_height=sat_height,
                                   false_easting=0, false_northing=0, globe=globe, sweep_axis=sat_sweep)

    crs_plt = ccrs.PlateCarree() # Globe keyword was messing everything up

    trans_pts = crs_geos.transform_points(crs_plt, np.array([x_min, x_max]), np.array([y_min, y_max]))

    proj_extent = (min(trans_pts[0][0], trans_pts[1][0]), max(trans_pts[0][0], trans_pts[1][0]),
                   min(trans_pts[0][1], trans_pts[1][1]), max(trans_pts[0][1], trans_pts[1][1]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([extent['min_lon'], extent['max_lon'], extent['min_lat'], extent['max_lat']], crs=crs_plt)

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                        name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=0)
    ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])
    ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])

    mrms_ref = np.memmap(mrms_obj.get_data_path(), dtype='float32', mode='r', shape=mrms_obj.shape)
    mrms_ref = np.asarray(mrms_ref)
    mrms_ref = mrms_ref.astype('float')
    mrms_ref[mrms_ref == 0] = np.nan

    refl = plt.pcolormesh(mrms_obj.grid_lons, mrms_obj.grid_lats, mrms_ref, transform=crs_plt,
                cmap=cm.gist_ncar, zorder=z_ord['mrms'])

    cbar_mrms = plt.colorbar(refl,fraction=0.046, pad=0.04)
    plt.setp(cbar_mrms.ax.yaxis.get_ticklabels(), fontsize=12)
    cbar_mrms.set_label('Reflectivity (dbz)', fontsize = 14, labelpad = 20)

    inf_img1 = plt.imshow(sat_data['data'], cmap=cm.Greys, origin='upper',
                                 vmin=190, vmax=290, extent=proj_extent, zorder=z_ord['sat'],
                                 alpha=0.8, transform=crs_geos)

    cbar_bounds = np.arange(190, 290, 10)
    cbar_sat = fig.colorbar(inf_img1, ticks=[x for x in cbar_bounds], spacing='proportional',
                        fraction=0.046, pad=0.02, shrink=0.53)
    cbar_sat.set_ticklabels([str(x) for x in cbar_bounds], update_ticks=True)
    cbar_sat.ax.tick_params(labelsize=6)
    cbar_sat.set_label('Cloud-top Temperature (K)', fontsize=8)

    if (range_rings):
        clrs = ['g', 'y']
        for idx, x in enumerate([100, 250]):
            coord_list = geodesic_point_buffer(cent_lat, cent_lon, x)
            lats = [float(x[1]) for x in coord_list.coords[:]]
            max_lat = max(lats)

            mpl_poly = Polygon(np.array(coord_list), ec=clrs[idx], fc="none", transform=crs_plt,
                                   linewidth=1.25, zorder=z_ord['map'])
            ax.add_patch(mpl_poly)

    if (wwa_polys is not None):
        wwa_keys = wwa_polys.keys()

        if ('SV' in wwa_keys):
            sv_polys = cfeature.ShapelyFeature(wwa_polys['SV'], ccrs.PlateCarree())
            ax.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=z_ord['wwa'])
        if ('TO' in wwa_keys):
            to_polys = cfeature.ShapelyFeature(wwa_polys['TO'], ccrs.PlateCarree())
            ax.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=z_ord['wwa'])

    if (points_to_plot is not None):
        plt.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='r', zorder=4, transform=ccrs.PlateCarree())

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

    ax.set_aspect('equal', adjustable='box')

    if (save):
        if (outpath is not None):
            fname = 'abi-mrms-{}-{}z.png'.format(mrms_obj.validity_date, mrms_obj.validity_time)
            path = join(outpath, fname)
            plt.savefig(path, dpi=500, bbox_inches='tight')
        else:
            raise ValueError('Error: Outpath cannot be None')
    if (show):
        plt.show()
    plt.close('all')



def plot_mrms_lma_abi_glm(sat_data, mrms_obj, glm_obj, wtlma_obj, grid_extent=None,
                points_to_plot=None, range_rings=False, wwa_polys=None, show=True,
                save=False, outpath=None):
    """
    Creates a figure with two side-by-side plots. The first (left) plot displays
    the ABI visual & infrared 'sandwhich' product with GLM FED overlay. The
    second (right) plot displays MRMS composite reflectivity with LMA overlay.

    Parameters
    ----------
    sat_data : tuple of LocalGoesFile objects
        Format: (visual, infrared)
    mrms_obj : MRMSGrib object
    glm_obj : LocalGLMFile
    wtlma_obj : LocalWTLMAFile
    grid_extent : dictionary
        Dictionary that defines the extent of the data grid
        Keys: min_lon, max_lon, min_lat, max_lat
    points_to_plot : tuple of tuples or list of tuples, optional
        Format: [(lat1, lon1), (lat2, lon2)]
    range_rings : bool, optional
        If true, plots color-coded WTLMA range-rings to indicate the possibly
        decrease in data quality due to distance
    wwa_polys : dict; key : str, value : polygon; optional
        NWS Severe Thunderstorm and/or Tornado warning polygons
    satellite_data: dict, optional
        Dictionary of satellite data & metadata to plot
    show : bool, optional
        If True, the plot will be displayed in the matplotlib GUI. Default is True
    save : bool, optional
        If True, the plot will be saved to the location specified by the outpath
        parameter, which must be specified. Default is False
    outpath : str, optional
        The path to the directory to save the plot. If save is True, outpath
        cannot be None
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    z_ord = {'map':6 , 'sat_vis': 1, 'sat_inf': 2, 'glm': 3, 'lma': 4, 'wwa': 5, 'top': 10}
    tx_counties_reader = shpreader.Reader(TX_SHP_PATH)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(OK_SHP_PATH)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    cent_lat = float(wtlma_obj.coord_center[0])
    cent_lon = float(wtlma_obj.coord_center[1])

    if (grid_extent is None):
        bounds = geodesic_point_buffer(cent_lat, cent_lon, 300)
        lats = [float(x[1]) for x in bounds.coords[:]]
        lons = [float(x[0]) for x in bounds.coords[:]]
        extent = {'min_lon': min(lons), 'max_lon': max(lons), 'min_lat': min(lats),
                    'max_lat': max(lats)}
        del lats
        del lons
    else:
        extent = grid_extent

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'],
                semiminor_axis=glm_obj.data['semi_minor_axis'],vflattening=None,
                inverse_flattening=glm_obj.data['inv_flattening'])

    crs_plt = ccrs.PlateCarree() # Globe keyword was messing everything up

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'],
                glm_obj.data['height'], glm_obj.data['sweep_ang_axis'])

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Mercator()},
                        figsize=(12, 8))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                        name='admin_1_states_provinces_shp', zorder=0)

    for ax in [ax1, ax2]:
        ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)
        ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray',
                    zorder=z_ord['map'])

        ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray',
                    zorder=z_ord['map'])

        ax.set_extent([extent['min_lon'], extent['max_lon'], extent['min_lat'],
                    extent['max_lat']], crs=crs_plt)

    grid_lons = np.arange(extent['min_lon'], extent['max_lon'], 0.01)
    grid_lats = np.arange(extent['min_lat'], extent['max_lat'], 0.01)

    ############################## Plot GLM data ##############################
    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = ax1.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=crs_plt,
                        cmap=cm.jet, zorder=z_ord['glm'])

    axins1 = inset_axes(ax1, width="5%", height="100%", loc='lower left',
                bbox_to_anchor=(1, 0., 1, 1), bbox_transform=ax1.transAxes, borderpad=0)

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional',
                        fraction=0.046, pad=0, cax=axins1)

    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label('GLM Flash Extent Density', fontsize=8)

    ############################## Plot LMA data ##############################
    lma_norm = colors.LogNorm(vmin=1, vmax=400)

    H, X_edges, Y_edges = np.histogram2d(wtlma_obj.data['lon'], wtlma_obj.data['lat'],
                          bins=100, range=[[extent['min_lon'], extent['max_lon']],
                          [extent['min_lat'], extent['max_lat']]],
                          weights=wtlma_obj.data['P']) # bins=[len(grid_lons), len(grid_lats)]

    lma_mesh = ax2.pcolormesh(X_edges, Y_edges, H.T, norm=lma_norm, transform=crs_plt,
                        cmap=cm.inferno, zorder=z_ord['lma'])

    lma_bounds = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400]

    axins2 = inset_axes(ax2, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0., 1, 1),
                    bbox_transform=ax2.transAxes, borderpad=0)

    cbar2 = plt.colorbar(lma_mesh, ticks=lma_bounds, spacing='proportional',fraction=0.046,
                    pad=0.02, cax=axins2)

    cbar2.ax.set_yticklabels([str(x) for x in lma_bounds])
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('WTLMA Source Power Density (dBW)', fontsize=8)

    ############################## Plot MRMS data ##############################
    mrms_ref = np.memmap(mrms_obj.get_data_path(), dtype='float32', mode='r',
                shape=mrms_obj.shape)
    mrms_ref = np.asarray(mrms_ref)
    mrms_ref = mrms_ref.astype('float')
    mrms_ref[mrms_ref == 0] = np.nan

    ref_plot = ax2.pcolormesh(mrms_obj.grid_lons, mrms_obj.grid_lats, mrms_ref,
                transform=crs_plt, cmap=cm.gist_ncar, zorder=z_ord['mrms'])

    cbar_mrms = fig.colorbar(ref_plot, spacing='proportional',
                        fraction=0.046, pad=0.02, shrink=0.53, ax=ax2,
                        orientation='horizontal')
    cbar_mrms.ax.tick_params(labelsize=6)
    cbar_mrms.set_label('MRMS Composite Reflectivity (dbz)', fontsize=8)

    ############################ Plot x-sect coords ############################
    if (points_to_plot is not None):
        ax1.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0],
                    points_to_plot[1][0]], marker='o', color='r', zorder=z_ord['wwa'],
                    transform=crs_plt)
        ax2.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0],
                    points_to_plot[1][0]], marker='o', color='r', zorder=z_ord['wwa'],
                    transform=crs_plt)

    ########################### Plot LMA Range Rings ###########################
    if (range_rings):
        clrs = ['g', 'y']
        for idx, x in enumerate([100, 250]):
            coord_list = geodesic_point_buffer(cent_lat, cent_lon, x)
            lats = [float(x[1]) for x in coord_list.coords[:]]
            max_lat = max(lats)

            # Only way mpl won't throw a fit about using the patch twice
            for ax in [ax1, ax2]:
                # https://stackoverflow.com/questions/27574897/plotting-disconnected-entities-
                # with-shapely-descartes-and-matplotlib
                mpl_poly = Polygon(np.array(coord_list), ec=clrs[idx], fc="none",
                            transform=crs_plt, linewidth=1.25, zorder=z_ord['map'])
                ax.add_patch(mpl_poly)

    ############################## Plot Sat data ##############################
    if (len(satellite_data) != 2):
        raise ValueError('Error: Invalid satellite data params to produce sandwhich image')
    else:
        visual = satellite_data[0]
        infrared = satellite_data[1]

        sat_height = visual['sat_height']
        sat_lon = visual['sat_lon']
        sat_sweep = visual['sat_sweep']
        scan_date = visual['scan_date']

        y_min, x_min = scan_to_geod(min(visual['y_image_bounds']), min(visual['x_image_bounds']))
        y_max, x_max = scan_to_geod(max(visual['y_image_bounds']), max(visual['x_image_bounds']))

        crs_geos = ccrs.Geostationary(central_longitude=sat_lon, satellite_height=sat_height,
                                       false_easting=0, false_northing=0, globe=globe,
                                       sweep_axis=sat_sweep)

        trans_pts = crs_geos.transform_points(crs_plt, np.array([x_min, x_max]),
                    np.array([y_min, y_max]))

        proj_extent = (min(trans_pts[0][0], trans_pts[1][0]),
                       max(trans_pts[0][0], trans_pts[1][0]),
                       min(trans_pts[0][1], trans_pts[1][1]),
                       max(trans_pts[0][1], trans_pts[1][1]))

        viz_img1 = ax1.imshow(visual['data'], cmap=cm.Greys_r, extent=proj_extent, origin='upper',
                             vmin=visual['min_data_val'], vmax=visual['max_data_val'],
                             zorder=z_ord['sat_vis'], transform=crs_geos)

        inf_img1 = ax1.imshow(infrared['data'], cmap=cm.nipy_spectral_r, origin='upper',
                             vmin=190, vmax=270, extent=proj_extent, zorder=z_ord['sat_inf'],
                             alpha=0.4, transform=crs_geos)


        cbar_bounds = np.arange(190, 270, 10)
        cbar_sat = fig.colorbar(inf_img1, ticks=[x for x in cbar_bounds], spacing='proportional',
                            fraction=0.046, pad=0.02, shrink=0.53, ax=ax1,
                            orientation='horizontal')
        cbar_sat.set_ticklabels([str(x) for x in cbar_bounds], update_ticks=True)
        cbar_sat.ax.tick_params(labelsize=6)
        cbar_sat.set_label('Cloud-top Temperature (K)', fontsize=8)

    ############################## Plot WWA Polys ##############################
    if (wwa_polys is not None):
        wwa_keys = wwa_polys.keys()

        if ('SV' in wwa_keys):
            sv_polys = cfeature.ShapelyFeature(wwa_polys['SV'], ccrs.PlateCarree())
            ax1.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow',
                        order=z_ord['wwa'])
            ax2.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow',
                        zorder=z_ord['wwa'])
        if ('TO' in wwa_keys):
            to_polys = cfeature.ShapelyFeature(wwa_polys['TO'], ccrs.PlateCarree())
            ax1.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red',
                        zorder=z_ord['wwa'])
            ax2.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red',
                        zorder=z_ord['wwa'])

    ax1.set_title('GLM Flash Extent Density {} {}z'.format(glm_obj.scan_date, glm_obj.scan_time),
                   loc='center', fontsize=8)
    ax2.set_title(('WTLMA Power-Weighted Source Density & MRMS Composite'
                   ' Reflectivity {}z').format(wtlma_obj._start_time_pp()),
                   loc='center', fontsize=8)

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    if (save):
        if (outpath is not None):
            fname = 'sbs2-plan-{}-{}z.png'.format(glm_obj.scan_date, glm_obj.scan_time)
            path = join(outpath, fname)
            plt.savefig(path, dpi=500, bbox_inches='tight')
        else:
            raise ValueError('Error: Outpath cannot be None')
    if (show):
        plt.show()
    plt.close('all')



def plot_cross_cubic_single(grb, point1, point2, first=False):
    """
    Plots the cross section of a single MRMSGrib object's data from point1 to point2
    using cubic interpolation

    Parameters
    ----------
    grb : MRMSGrib object
    point1 : tuple of float
        Coordinates of the first point that defined the cross section
        Format: (lat, lon)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lat, lon)
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
    y_world, x_world = np.array(list(zip(*line)))
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
        Format: (lat, lon)
    point2 : tuple of float
        Coordinates of the second point that defined the cross section
        Format: (lat, lon)
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
    y_world, x_world = np.array(list(zip(*line)))
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
            lats = grib.trunc(lats, 2)
            lons = grib.trunc(lons, 2)
            coords = []
            for idx, x in enumerate(lons):
                coords.append('(' + str(x) + ', ' + str(lats[idx]) + ')')
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
    im = ax.pcolormesh(coords, scan_angles, data, cmap=mpl.cm.gist_ncar, vmin=0, vmax=65)
    cbar = fig.colorbar(im, ax=ax, ticks=[10,20,30,40,50,60])
    cbar.set_label('Reflectivity (dbz)', rotation=90)
    ax.set_title('MRMS Reflectivity Cross Section')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_ylabel('Scan Angle (Deg)')
    ax.set_xlabel('Lon, Lat')

    fig.tight_layout()

    plt.show()



def plot_mrms_cross_section2(data=None, abs_path=None, lons=None, lats=None, wtlma_obj=None,
            wtlma_coords=None, show=True, save=False, outpath=None):
    """
    Plots a cross-section of MRMS reflectivity data from all scan angles, with WTLMA
    events overlayed. If the 'data' parameter is given, then that data is plotted.
    If 'abs_path' is given, then data from the text file located at that absolute
    path is read and plotted

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
            lats = grib.trunc(lats, 2)
            lons = grib.trunc(lons, 2)
            coords = []
            for idx, x in enumerate(lons):
                coords.append('(' + str(x) + ', ' + str(lats[idx]) + ')')
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

    if (wtlma_obj is None):
        raise ValueError('Missing wtlma_obj param')

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    ax.set_ylim([500, 19000]) # 500 to 19000 meters

    ref_norm = colors.Normalize(vmin=0, vmax=65)

    if (len(coords) != data.shape[1]):
        print('WARNING: Width of data array != length of x-axis values')

    im = ax.pcolormesh(coords, scan_angles*1000, data, cmap=mpl.cm.gist_ncar, vmin=0, vmax=65)
    cbar = fig.colorbar(im, ax=ax, ticks=[10,20,30,40,50,60])
    cbar.set_label('Reflectivity (dbz)', rotation=90)

    wtlma_lats, wtlma_lons = list(zip(*wtlma_coords))

    wtlma_lats = list(wtlma_lats)
    wtlma_lons = list(wtlma_lons)
    wtlma_lats = grib.trunc(wtlma_lats, 2)
    wtlma_lons = grib.trunc(wtlma_lons, 2)

    wtlma_coords = []
    for idx, x in enumerate(wtlma_lons):
        wtlma_coords.append('(' + str(x) + ', ' + str(wtlma_lats[idx]) + ')')

    scatt = ax.scatter(wtlma_coords, wtlma_obj.data['alt'], c=wtlma_obj.data['P'],
                       marker='o', s=100, cmap=mpl.cm.jet, vmin=-20, vmax=100, alpha=0.5, zorder=2)

    cbar2 = fig.colorbar(scatt, ax=ax)
    cbar2.set_label('WTLMA Stroke Power (dBW)', rotation=90)

    ax.set_title('MRMS Reflectivity Cross Section with WTLMA Sources < 19000m AGL\n{}'.format(wtlma_obj._start_time_pp()), loc='left')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_rotation(30)
        tick.label.set_horizontalalignment("right")

    ax.set_ylabel('Altitude (m)')
    ax.set_xlabel('Lon, Lat')
    ax.margins(0, tight=True)

    fig.tight_layout()

    if (save):
        if (outpath):
            fname = 'xsect-{}z.png'.format(wtlma_obj.start_time)
            path = join(outpath, fname)
            plt.savefig(path, dpi=500)
        else:
            raise ValueError('Error: Outpath cannot be None')
    if (show):
        plt.show()
    plt.close('all')



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



def plot_wtlma(wtlma_obj_list, grid_extent=None, nbins=1000, points_to_plot=None):
    """
    Plots WTLMA data as a colormesh

    Parameters
    ----------
    wtlma_obj_list : list of LocalWTLMAFile objects
    grid_extent : Dictionary, optional
        Defines the extent of the grid
        Keys: min_lat, max_lat, min_lon, max_lon
    nbins : int, optional
        Number of bins to use when computing the 2D histogram. Default is 1000
    points_to_plot : list of tuples, optional
        Coordinate pairs to plot
        Format: [(lat1, lon1), (lat2, lon2)]
    """
    if (not isinstance(wtlma_obj_list, list)):
        wtlma_obj_list = [wtlma_obj_list]

    cent_lat = float(wtlma_obj_list[0].coord_center[0])
    cent_lon = float(wtlma_obj_list[0].coord_center[1])

    wtlma_data = wtlma_obj_list[0].data

    for obj in wtlma_obj_list[1:]:
        wtlma_data = wtlma_data.append(obj.data)

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator())

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=1)

    if (grid_extent is None):
        bounds = geodesic_point_buffer(cent_lat, cent_lon, 300)
        lats = [float(x[1]) for x in bounds.coords[:]]
        lons = [float(x[0]) for x in bounds.coords[:]]
        extent = {'min_lon': min(lons), 'max_lon': max(lons), 'min_lat': min(lats), 'max_lat': max(lats)}
        del lats
        del lons
    else:
        extent = grid_extent

    ax.set_extent([extent['min_lon'], extent['max_lon'], extent['min_lat'], extent['max_lat']], crs=ccrs.PlateCarree())

    grid_lons = np.arange(extent['min_lon'], extent['max_lon'], 0.01)
    grid_lats = np.arange(extent['min_lat'], extent['max_lat'], 0.01)

    lma_norm = colors.LogNorm(vmin=1, vmax=150)

    H, X_edges, Y_edges = np.histogram2d(wtlma_data['lon'], wtlma_data['lat'],
                          bins=nbins, range=[[extent['min_lon'], extent['max_lon']], [extent['min_lat'], extent['max_lat']]])

    lma_mesh = ax.pcolormesh(X_edges, Y_edges, H.T, norm=lma_norm, transform=ccrs.PlateCarree(), cmap=cm.inferno, zorder=3)
    lma_bounds = [5, 10, 15, 20, 25, 50, 100, 150]
    cbar = plt.colorbar(lma_mesh, ticks=lma_bounds, spacing='proportional',fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([str(x) for x in lma_bounds])
    cbar.set_label('WTLMA Source Density')

    if (points_to_plot is not None):
        plt.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='r', zorder=4, transform=ccrs.PlateCarree())

    for x in [100, 250]:
        coord_list = geodesic_point_buffer(cent_lat, cent_lon, x)
        lats = [float(x[1]) for x in coord_list.coords[:]]
        max_lat = max(lats)

        # https://stackoverflow.com/questions/27574897/plotting-disconnected-entities-with-shapely-descartes-and-matplotlib
        mpl_poly = Polygon(np.array(coord_list), ec="r", fc="none", transform=ccrs.PlateCarree(),
                           linewidth=1.25, zorder=2)
        ax.add_patch(mpl_poly)
        plt.text(cent_lon, max_lat + 0.05, str(x) + " km", color = "r", horizontalalignment="center", transform=ccrs.PlateCarree(),
                 fontsize = 15)

    if (len(wtlma_obj_list) == 1):
        plt.title('WTLMA Flashes {}'.format(wtlma_obj_list[0]._start_time_pp()), loc='right')
    else:
        plt.title('WTLMA Flashes {} to {}'.format(wtlma_obj_list[0]._data_start_pp(), wtlma_obj_list[-1]._data_end_pp()), loc='right')
    #plt.tight_layout()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def plot_mrms_glm(grb_obj, glm_obj, wtlma_obj=None, points_to_plot=None, wwa_polys=None):
    """
    Plots MRMS and GLM data on a Mercator projection

    Parameters
    ----------
    grb_obj : MRMSGrib object
        MRMSGrib object containing the MRMS data to plot
    glm_obj : LocalGLMFile object
        LocalGLMFile object containing the GLM data to plot
    wtlma_obj : LocalWTLMAFile, optional
        WTLMA object to plot
    points_to_plot : tuple of tuples or list of tuples, optional
        Format: [(lat1, lon1), (lat2, lon2)]
    wwa_polys : dict; key : str, value : polygon; optional
        NWS Severe Thunderstorm and/or Tornado warning polygons

    Notes
    -----
    Axis extent set by subsetted MRMS lat & lon extent
    """
    z_ord = {'map': 1, 'mrms': 2, 'sat_vis': 3, 'sat_inf': 4, 'glm': 5, 'lma': 6, 'wwa': 7, 'top': 10}

    tx_counties_reader = shpreader.Reader(TX_SHP_PATH)
    tx_counties_list = list(tx_counties_reader.geometries())
    tx_counties = cfeature.ShapelyFeature(tx_counties_list, ccrs.PlateCarree())

    ok_counties_reader = shpreader.Reader(OK_SHP_PATH)
    ok_counties_list = list(ok_counties_reader.geometries())
    ok_counties = cfeature.ShapelyFeature(ok_counties_list, ccrs.PlateCarree())

    globe = ccrs.Globe(semimajor_axis=glm_obj.data['semi_major_axis'], semiminor_axis=glm_obj.data['semi_minor_axis'],
                       flattening=None, inverse_flattening=glm_obj.data['inv_flattening'])

    Xs, Ys = georeference(glm_obj.data['x'], glm_obj.data['y'], glm_obj.data['lon_0'], glm_obj.data['height'],
                          glm_obj.data['sweep_ang_axis'])

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='black',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='gray', zorder=z_ord['map'])
    ax.add_feature(tx_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])
    ax.add_feature(ok_counties, linewidth=.6, facecolor='none', edgecolor='gray', zorder=z_ord['map'])

    ax.set_extent([min(grb_obj.grid_lons), max(grb_obj.grid_lons), min(grb_obj.grid_lats), max(grb_obj.grid_lats)], crs=ccrs.PlateCarree())

    mrms_ref = np.memmap(grb_obj.get_data_path(), dtype='float32', mode='r', shape=grb_obj.shape)
    mrms_ref = np.asarray(mrms_ref)
    mrms_ref = mrms_ref.astype('float')
    mrms_ref[mrms_ref == 0] = np.nan

    refl = plt.pcolormesh(grb_obj.grid_lons, grb_obj.grid_lats, mrms_ref, transform=ccrs.PlateCarree(), cmap=cm.gist_ncar, zorder=z_ord['mrms'])

    bounds = [5, 10, 20, 50, 100, 150, 200, 300, 400]
    glm_norm = colors.LogNorm(vmin=1, vmax=max(bounds))

    cmesh = plt.pcolormesh(Xs, Ys, glm_obj.data['data'], norm=glm_norm, transform=ccrs.PlateCarree(), cmap=cm.jet, zorder=z_ord['glm'])
    cbar2 = plt.colorbar(refl,fraction=0.046, pad=0.04)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), fontsize=12)
    cbar2.set_label('Reflectivity (dbz)', fontsize = 14, labelpad = 20)

    cbar1 = plt.colorbar(cmesh, norm=glm_norm, ticks=bounds, spacing='proportional', fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels([str(x) for x in bounds])
    cbar1.set_label('GLM Flash Extent Density')

    if (wtlma_obj is not None):
        scat = plt.scatter(wtlma_obj.data['lon'], wtlma_obj.data['lat'], c='gray',
                           marker='o', s=50, zorder=z_ord['lma'], transform=ccrs.PlateCarree(), alpha=0.5)

    if (points_to_plot is not None):
        plt.plot([points_to_plot[0][1], points_to_plot[1][1]], [points_to_plot[0][0], points_to_plot[1][0]],
                           marker='o', color='black', zorder=z_ord['top'], transform=ccrs.PlateCarree())

    if (wwa_polys is not None):
        wwa_keys = wwa_polys.keys()

        if ('SV' in wwa_keys):
            sv_polys = cfeature.ShapelyFeature(wwa_polys['SV'], ccrs.PlateCarree())
            ax.add_feature(sv_polys, linewidth=.8, facecolor='none', edgecolor='yellow', zorder=z_ord['wwa'])
        if ('TO' in wwa_keys):
            to_polys = cfeature.ShapelyFeature(wwa_polys['TO'], ccrs.PlateCarree())
            ax.add_feature(to_polys, linewidth=.8, facecolor='none', edgecolor='red', zorder=z_ord['wwa'])

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

    # Increase font size of colorbar tick labels
    plt.title('MRMS Reflectivity ' + str(grb_obj.validity_date) + ' ' + str(grb_obj.validity_time) + 'z')

    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    #fig.savefig(join(out_path, scan_date.strftime('%Y'), scan_date.strftime('%Y%m%d-%H%M')) + '.png', dpi=500)

    plt.show()



def run_mrms_xsect(base_path, slice_time, point1, point2):
    """
    Preforms some function calls needed to execute plot_mrms_cross_section()

    Parameters
    ----------
    base_path : str
        Path to the parent MRMS data directory
    slice_time : str
        Validity time of the MRMS data
    point1 : tuple of floats
        First point defining the cross section
    point2 : tuple of floats
        Second point defining the cross section
    """
    #f_out = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'

    cross_data, lats, lons = process_slice(base_path, slice_time, point1, point2)
    plot_mrms_cross_section(data=cross_data, lons=lons, lats=lats)



def run_mrms_xsect2(base_path, slice_time, point1, point2, wtlma_obj, wtlma_coords,
        show=False, save=False, outpath=None):
    """
    Preforms some function calls needed to execute plot_mrms_cross_section2()

    Parameters
    ----------
    base_path : str
        Path to the parent MRMS data directory
    slice_time : str
        Validity time of the MRMS data
    point1 : tuple of floats
        First point defining the cross section
    point2 : tuple of floats
        Second point defining the cross section
    wtlma_obj : LocalWTLMAFile obj
        Object containing WTLMA data
    wtlma_coords : list of tuple
        List of coordinates of filtered WTLMA events (?)
        Format: (lat, lon)
    """
    print('Warning: Depricated')
    cross_data, lats, lons = process_slice(base_path, slice_time, point1, point2)
    plot_mrms_cross_section2(data=cross_data, lons=lons, lats=lats, wtlma_obj=wtlma_obj,
            wtlma_coords=wtlma_coords, show=show, save=save, outpath=outpath)



def run_mrms_xsect_inset(base_path, slice_time, point1, point2):
    """
    Preforms some function calls needed to execute plot_cross_section_inset()

    Parameters
    ----------
    base_path : str
        Path to the parent MRMS data directory
    slice_time : str
        Validity time of the MRMS data
    point1 : tuple of floats
        First point defining the cross section
    point2 : tuple of floats
        Second point defining the cross section
    """
    print('Warning: Depricated')
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    f_out = '/media/mnichol3/pmeyers1/MattNicholson/mrms/x_sect'

    f_dict = process_slice_inset(base_path, '2124', point1, point2)
    plot_cross_section_inset(inset_data=f_dict['f_inset_data'], inset_lons=f_dict['f_inset_lons'],
                             inset_lats=f_dict['f_inset_lats'], abs_path=f_dict['x_sect'], points=(point1, point2))
