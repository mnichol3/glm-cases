"""
Author: Matt Nicholson

"""
import sys
import six
import pandas as pd
from sys import exit

import wtlma
import goesawsinterface
import grib
import localglminterface
import glm_utils
import plotting_funcs
import plotting_utils



def make_wtlma_plot(base_path, start, stop, points_to_plot=None):
    # '05-23-2019-21:00', '05-23-2019-21:55'
    lma_objs = []
    wtlma_files = wtlma.get_files_in_range(local_wtlma_path, start, stop)

    for f in wtlma_files:
        abs_path = wtlma._parse_abs_path(local_wtlma_path, f)
        lma_objs.append(wtlma.parse_file(abs_path))
    plotting_funcs.plot_wtlma(lma_objs, points_to_plot=points_to_plot)



def make_mrms_glm_plot(local_mrms_path, local_glm_path, date, time, point1, point2):
    mrms_scans = grib.fetch_scans(local_mrms_path, time)
    mrms_obj = grib.get_grib_objs(mrms_scans[12], local_mrms_path, point1, point2)[0]

    del mrms_scans

    t1 = _format_date_time(date, time)
    glm_scans = localglminterface.get_files_in_range(local_glm_path, t1, t1)
    # 2 files for each time, apparently from 2 diff sources but same data
    glm_obj = glm_utils.read_file(glm_scans[0].abs_path, meta=True)
    plotting_funcs.plot_mrms_glm(mrms_obj, glm_obj)



def main():
    local_abi_path = '/media/mnichol3/pmeyers1/MattNicholson/goes'
    local_wtlma_path = '/media/mnichol3/pmeyers1/MattNicholson/wtlma'
    local_glm_path = '/media/mnichol3/pmeyers1/MattNicholson/glm'
    local_mrms_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'

    abs_path_glm = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523/IXTR99_KNES_232121_40312.2019052321'
    abs_path_wtlma = '/media/mnichol3/pmeyers1/MattNicholson/wtlma/2019/05/23/LYLOUT_190523_212000_0600.dat'

    case_coords = '/home/mnichol3/Coding/glm-cases/resources/05232019-coords.txt'
    d_dict = {'date': str, 'wsr-time': str, 'mrms-time': str, 'lat1': float,
              'lon1': float, 'lat2': float, 'lon2': float}
    case_steps = pd.read_csv(case_coords, sep=',', header=0, dtype=d_dict)


    for idx, step in case_steps.iterrows():
        point1 = (step['lat1'], step['lon1'])
        point2 = (step['lat2'], step['lon2'])

        point1 = grib.trunc(point1, 3)
        point2 = grib.trunc(point2, 3)

        make_mrms_glm_plot(local_mrms_path, local_glm_path, step['date'], step['mrms-time'], point1, point2)


    """
    abs_path_wtlma = '/media/mnichol3/pmeyers1/MattNicholson/wtlma/2019/05/23/LYLOUT_190523_211000_0600.dat'
    wtlma_data = wtlma.parse_file(abs_path_wtlma, sub_t='21:19')
    filtered_data, coords = plotting_utils.filter_by_dist(wtlma_data.data, 1000, point1, point2, 100)
    #print(len(filtered_data['lon']))
    #print(len(coords))
    #sys.exit(0)
    wtlma_data._set_data(filtered_data)
    plotting_funcs.run_mrms_xsect2(local_mrms_path, '2119', point1, point2, wtlma_data, coords)
    """

    """
    mrms_scans = grib.fetch_scans(local_mrms_path, '2106')
    mrms_obj = grib.get_grib_objs(mrms_scans[12], local_mrms_path, point1, point2)[0]

    del mrms_scans

    glm_scans = localglminterface.get_files_in_range(local_glm_path, '05-23-2019-21:06','05-23-2019-21:06')
    glm_obj = glm_utils.read_file(glm_scans[0].abs_path, meta=True)

    plotting_funcs.plot_mrms_glm(mrms_obj, glm_obj)
    """


    """
    glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=False)
    wtlma_data = wtlma.parse_file(abs_path_wtlma, sub_t='21:21')
    plotting_funcs.plot_mercator_dual_2(glm_data, wtlma_data, points_to_plot=(point1, point2), range_rings=True)
    """


    #cross_data, lats, lons = plotting_utils.process_slice(local_mrms_path, '2119', point1, point2)
    #plotting_funcs.plot_mrms_cross_section2(data=cross_data, lons=lons, lats=lats, wtlma_df=wtlma_data.data)
    #lma_extent = {'min_lon': -101.365, 'max_lon': -101.115, 'min_lat': 35.565, 'max_lat': 36.045}


    #plotting_funcs.plot_mrms_cross_section2(data=None, abs_path=None, lons=None, lats=None, wtlma_df=None)
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)

    #plotting_funcs.plot_mercator_dual(glm_data, (point1, point2), wtlma_data)
    #plotting_funcs.plot_mercator_dual_2(glm_data, (point1, point2), wtlma_data)


    """
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    plotting_funcs.run_mrms_xsect(base_path, '2124', point1, point2)
    """

    """
    conn = goesawsinterface.GoesAWSInterface()
    imgs = conn.get_avail_images_in_range('goes16', 'ABI-L2-CMIPM', '5-23-2019-20:00', '5-23-2019-21:00', 'M1', '13')
    for x in imgs:
        print(x)

    avail_glm_imgs = localglminterface.get_files_in_range('/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523', '5-23-2019-20:00', '5-23-2019-21:00')
    for x in avail_glm_imgs:
        print(x)
    """


def _format_date_time(date, time):
    month = date[:2]
    day = date[2:4]
    year = date[-4:]

    hour = time[:2]
    mint = time[2:]
    return '{}-{}-{}-{}:{}'.format(month, day, year, hour, mint)


if (__name__ == '__main__'):
    main()
