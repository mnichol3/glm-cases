from grib import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from wrf import to_np, getvar, CoordPair, vertcross
import wrf
from os.path import join



def plot_cross(data, z):

    # Create the start point and end point for the cross section
    start_point = CoordPair(lat=37.195, lon=-102.185)
    end_point = CoordPair(lat=34.565, lon=-99.865)
    ll_point = CoordPair(lat=34.565, lon=-102.185)

    # Compute the vertical cross-section interpolation.  Also, include the
    # lat/lon points along the cross-section.
    ref_cross = vertcross(data, z, projection=wrf.Mercator, start_point=start_point,
                            end_point=end_point, ll_point=ll_point, latlon=True, meta=True)

    # Create the figure
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()

    # Make the contour plot
    ref_contours = ax.contourf(to_np(ref_cross), cmap=get_cmap("jet"))

    # Add the color bar
    plt.colorbar(ref_contours, ax=ax)

    # Set the x-ticks to use latitude and longitude labels.
    coord_pairs = to_np(ref_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [pair.latlon_str(fmt="{:.2f}, {:.2f}")
                for pair in to_np(coord_pairs)]
    ax.set_xticks(x_ticks[::20])
    ax.set_xticklabels(x_labels[::20], rotation=45, fontsize=8)

    # Set the y-ticks to be height.
    #vert_vals = to_np(ref_cross.coords["vertical"])
    v_ticks = np.arange(vert_vals.shape[0])
    ax.set_yticks(v_ticks[::20])
    #ax.set_yticklabels(vert_vals[::20], fontsize=8)

    # Set the x-axis and  y-axis labels
    ax.set_xlabel("Latitude, Longitude", fontsize=12)
    ax.set_ylabel("Scan Angle (deg)", fontsize=12)

    plt.title("Vertical Cross Section of Reflectivity (dbz)")

    plt.show()



def main():

    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms'
    #f_name = 'MRMS_MergedReflectivityQC_00.50_20190523-212434.grib2'
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    #f_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905/MergedReflectivityQC_01.50'
    #f_name = 'MRMS_MergedReflectivityQC_01.50_20190523-212434.grib2'
    #f_abs = join(f_path, f_name)

    scans = fetch_scans(base_path, '2124') # z = 33


    grib_files = get_grib_objs(scans, base_path)
    print(grib_files)

    data = [x.data for x in grib_files]
    data_3d = np.stack(data)

    plot_cross(data_3d, 33)






if (__name__ == '__main__'):
    main()
