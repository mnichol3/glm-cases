import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from cartopy.feature import NaturalEarthFeature
from pyproj import Proj, transform

points = []



def inv_transform(coords):
    x_in = coords[0]
    y_in = coords[1]


    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4326')
    x_out, y_out = transform(inProj, outProj, x_in, y_in)

    return [x_out, y_out]



def plot_grb_interactive(grb):
    """
    Takes a MRMSGrib object and plots it on a Mercator projection

    Parameters
    ----------
    grb : MRMSGrib object

    Returns
    -------
    None
    """
    def onclick(event):
        global points
        if (points == []):
            points.append([event.xdata, event.ydata])
            plt.plot(event.xdata, event.ydata, 'ro-')
            fig.canvas.draw()
        elif (len(points) >= 2):
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        else:
            points.append([event.xdata, event.ydata])
            xs, ys = list(zip(*points))
            plt.plot(xs, ys, 'ro-')
            fig.canvas.draw()

        return points


    fig = plt.figure(figsize=(8, 6)) #dpi = 200

    globe = ccrs.Globe(semimajor_axis=grb.major_axis, semiminor_axis=grb.minor_axis,
                       flattening=None)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(globe=globe))

    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                             name='admin_1_states_provinces_shp')

    ax.add_feature(states, linewidth=.8, edgecolor='black')

    ax.set_extent([min(grb.grid_lons), max(grb.grid_lons), min(grb.grid_lats), max(grb.grid_lats)], crs=ccrs.PlateCarree())

    cmesh = plt.pcolormesh(grb.grid_lons, grb.grid_lats, grb.data, transform=ccrs.PlateCarree(), cmap=cm.gist_ncar)

    lon_ticks = [x for x in range(-180, 181)]
    lat_ticks = [x for x in range(-90, 91)]

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

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


    trans_points = []
    inv = ax.transData.inverted()
    for x in points:
        #p_inv = inv.transform(x)
        trans_points.append(inv_transform(x))
    return trans_points


    #return points
