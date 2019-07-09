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



def make_mrms_xsect2(local_mrms_path, local_wtlma_path, date, time, point1, point2):
    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)
    filtered_data, coords = plotting_utils.filter_by_dist(wtlma_data.data, 1000, point1, point2, 100)
    wtlma_data._set_data(filtered_data)

    plotting_funcs.run_mrms_xsect2(local_mrms_path, time, point1, point2, wtlma_data, coords)


"""
def make_wtlma_glm_mercator_dual():
    glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=False)
    wtlma_data = wtlma.parse_file(abs_path_wtlma, sub_t='21:21')
    plotting_funcs.plot_mercator_dual_2(glm_data, wtlma_data, points_to_plot=(point1, point2), range_rings=True)
"""



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

        dt = _format_date_time(step['date'], step['mrms-time'])
        files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
        path = wtlma._parse_abs_path(local_wtlma_path, files[0])

        exit(0)
        #make_mrms_glm_plot(local_mrms_path, local_glm_path, step['date'], step['mrms-time'], point1, point2)


    
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)

    #plotting_funcs.plot_mercator_dual(glm_data, (point1, point2), wtlma_data)
    #plotting_funcs.plot_mercator_dual_2(glm_data, (point1, point2), wtlma_data)


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



def _format_time_wtlma(time):
    hr = time[:2]
    mn = time[2:]
    return '{}:{}'.format(hr, mn)


if (__name__ == '__main__'):
    main()
