from plot_cross import *
from interactive import *
from mrmsgrib import MRMSGrib

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from cartopy.feature import NaturalEarthFeature

grb_obj = get_grib_objs(['MRMS_MergedReflectivityQC_02.00_20190523-212434.grib2'], '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905')
points = plot_grb_interactive(grb_obj[0])
print(points)
