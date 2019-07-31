"""
Author: Matt Nicholson

"""
import sys
import six
import pandas as pd
from sys import exit, getrefcount
from datetime import datetime

import wtlma
import goesawsinterface
import grib
import localglminterface
import glm_utils
import plotting_funcs
import plotting_utils
import goes_utils



def make_wtlma_plot(base_path, start, stop, points_to_plot=None):
    """
    Contains helper-function calls needed for plot_wtlma()

    Parameters
    ----------
    base_path : str
        Path the the parent directory of the WTLMA data
    start : str
        Beginning of the data accumulation period
        Format: MM-DD-YYYY-HH:MM
    end : str
        End of the data accumulation period
        Format: MM-DD-YYYY-HH:MM
    points_to_plot : list of tuple, optional
        List of coordinates to plot, used to illustrate the location of the
        MRMS cross section.
        Format: [(lat1, lon1), (lat2, lon2)]
    """
    # '05-23-2019-21:00', '05-23-2019-21:55'
    lma_objs = []
    wtlma_files = wtlma.get_files_in_range(local_wtlma_path, start, stop)

    for f in wtlma_files:
        abs_path = wtlma._parse_abs_path(local_wtlma_path, f)
        lma_objs.append(wtlma.parse_file(abs_path))
    plotting_funcs.plot_wtlma(lma_objs, points_to_plot=points_to_plot)



def make_mrms_glm_plot(local_mrms_path, local_glm_path, local_wtlma_path, date,
                       time, point1, point2, memmap_path, wwa_fname):
    """
    Contains helper-function calls needed for plot_mrms_glm()

    Parameters
    ----------
    local_mrms_path : str
        Path the the parent directory of the MRMS data
    local_wtlma_path : str
        Path the the parent directory of the WTLMA data
    date : str
        Format: MMDDYYYY
    time : str
        Format: HHMM
    point1 : tuple of floats
        First point defining the cross section
    point2 : tuple of floats
        Second point defining the cross section
    """
    #mrms_scans = grib.fetch_scans(local_mrms_path, time)
    #mrms_obj = grib.get_grib_objs(mrms_scans[12], local_mrms_path, point1, point2)[0]
    #del mrms_scans
    mrms_obj = plotting_utils.get_composite_ref(local_mrms_path, time, point1, point2, memmap_path)

    t1 = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    wtlma_files = wtlma.get_files_in_range(local_wtlma_path, t1, t1)
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, wtlma_files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)

    glm_scans = localglminterface.get_files_in_range(local_glm_path, t1, t1)
    # 2 files for each time, apparently from 2 diff sources but same data
    glm_obj = glm_utils.read_file(glm_scans[1].abs_path, meta=True)
    wwa_polys = plotting_utils.get_wwa_polys(wwa_fname, date, time, wwa_type=['SV', 'TO'])
    # func sig : plot_mrms_glm(grb_obj, glm_obj, wtlma_obj=None, points_to_plot=None, wwa_polys=None)
    plotting_funcs.plot_mrms_glm(mrms_obj, glm_obj, wtlma_obj=wtlma_data,
            points_to_plot=[point1, point2], wwa_polys=wwa_polys)



def make_mrms_xsect2(local_mrms_path, local_wtlma_path, date, time, point1, point2,
                show=False, save=False, outpath=None):
    """
    Contains helper-function calls needed for run_mrms_xsect2()

    Parameters
    ----------
    local_mrms_path : str
        Path the the parent directory of the MRMS data
    local_wtlma_path : str
        Path the the parent directory of the WTLMA data
    date : str
        Format: MMDDYYYY
    time : str
        Format: HHMM
    point1 : tuple of floats
        First point defining the cross section
    point2 : tuple of floats
        Second point defining the cross section
    """
    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)
    # filter_by_dist(lma_df, dist, start_point, end_point, num_pts) dist in m
    filtered_data, coords = plotting_utils.filter_by_dist(wtlma_data.data, 4000, point1, point2, 100)
    wtlma_data._set_data(filtered_data)

    plotting_funcs.run_mrms_xsect2(local_mrms_path, time, point1, point2, wtlma_data, coords,
            show=show, save=save, outpath=outpath)



def make_wtlma_glm_mercator_dual(local_wtlma_path, local_glm_path, date, time,
         point1, point2, wwa_fname, sat_data, func, plot_extent, window=False,
         show=True, save=False, outpath=None):

    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    # grid_extent = None
    grid_extent = {'min_lat': plot_extent[0], 'max_lat': plot_extent[1],
                   'min_lon': plot_extent[2], 'max_lon': plot_extent[3]}

    glm_scans = localglminterface.get_files_in_range(local_glm_path, dt, dt)

    print('GLM 5-min window: {}'.format(window))
    print('GLM metadata: {} {}z {}'.format(glm_scans[0].scan_date, glm_scans[0].scan_time,
                glm_scans[0].filename))
    # 2 files for each time, apparently from 2 diff sources but same data
    glm_data = glm_utils.read_file(glm_scans[0].abs_path, meta=True, window=window)

    print('Fetching LMA data...')
    files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)

    print('Fetching WWA Polygons...')
    wwa_polys = plotting_utils.get_wwa_polys(wwa_fname, date, time, wwa_type=['SV', 'TO'])

    print('Plotting...')
    if (func == 1):
        plotting_funcs.plot_mercator_dual(glm_data, wtlma_data, points_to_plot=(point1, point2),
                                            range_rings=True, wwa_polys=wwa_polys,
                                            satellite_data=sat_data, grid_extent=grid_extent,
                                            show=show, save=save, outpath=outpath)
    elif (func == 2):
        plotting_funcs.plot_mercator_dual_2(glm_data, wtlma_data, points_to_plot=(point1, point2),
                                            range_rings=True, wwa_polys=wwa_polys,
                                            satellite_data=sat_data, grid_extent=grid_extent,
                                            show=show, save=save, outpath=outpath)
    else:
        raise ValueError('Invalid func param, must be 1 or 2')



def make_wtlma_glm_mercator_dual_hitemp(local_wtlma_path, local_glm_path, date, time,
         wwa_fname, sat_data, func, plot_extent, window=False, show=True, save=False, outpath=None):

    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    # grid_extent = None
    grid_extent = {'min_lat': plot_extent[0], 'max_lat': plot_extent[1],
                   'min_lon': plot_extent[2], 'max_lon': plot_extent[3]}

    glm_scans = localglminterface.get_files_in_range(local_glm_path, dt, dt)
    print('GLM 5-min window: {}'.format(window))
    print('GLM metadata: {} {}z {}'.format(glm_scans[0].scan_date, glm_scans[0].scan_time,
                glm_scans[0].filename))
    # 2 files for each time, apparently from 2 diff sources but same data
    glm_data = glm_utils.read_file(glm_scans[0].abs_path, meta=True, window=window)

    print('Fetching LMA data...')
    files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)

    print('Fetching WWA Polygons...')
    wwa_polys = plotting_utils.get_wwa_polys(wwa_fname, date, time, wwa_type=['SV', 'TO'])

    print('Plotting...')
    if (func == 1):
        plotting_funcs.plot_mercator_dual(glm_data, wtlma_data, points_to_plot=None,
                                            range_rings=True, wwa_polys=wwa_polys,
                                            satellite_data=sat_data, grid_extent=grid_extent,
                                            show=show, save=save, outpath=outpath)
    elif (func == 2):
        plotting_funcs.plot_mercator_dual_2(glm_data, wtlma_data, points_to_plot=None,
                                            range_rings=True, wwa_polys=wwa_polys,
                                            satellite_data=sat_data, grid_extent=grid_extent,
                                            show=show, save=save, outpath=outpath)
    else:
        raise ValueError('Invalid func param, must be 1 or 2')




def main():
    ############################ DEBUG ####################################
    # from pympler.tracker import SummaryTracker
    # tracker =  SummaryTracker()
    #######################################################################

    ############################# Data Paths ##############################

    ## Pat's drive
    local_abi_path = '/media/mnichol3/pmeyers1/MattNicholson/abi'
    local_wtlma_path = '/media/mnichol3/pmeyers1/MattNicholson/wtlma'
    local_glm_path = '/media/mnichol3/pmeyers1/MattNicholson/glm'
    local_mrms_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'
    img_outpath = '/home/mnichol3/Coding/glm-cases/imgs/05232019/auto-out'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## My drive
    # local_abi_path = '/media/mnichol3/tsb1/data/abi'
    # local_wtlma_path = '/media/mnichol3/tsb1/data/wtlma'
    # local_glm_path = '/media/mnichol3/tsb1/data/glm'
    # local_mrms_path = '/media/mnichol3/tsb1/data/mrms/201905'
    # memmap_path = '/media/mnichol3/tsb1/data/data'
    # img_outpath = '/home/mnichol3/Coding/glm-cases/imgs/05232019/auto-out'
    #######################################################################


    wwa_fname = ('/home/mnichol3/Coding/glm-cases/resources/wwa_201905230000_201905240000'
                 '/wwa_201905230000_201905240000.shp')
    case_coords = '/home/mnichol3/Coding/glm-cases/resources/05232019-coords.txt'
    d_dict = {'date': str, 'wsr-time': str, 'mrms-time': str, 'lat1': float,
              'lon1': float, 'lat2': float, 'lon2': float}
    case_steps = pd.read_csv(case_coords, sep=',', header=0, dtype=d_dict)


    ######################## Mercator Hi-Temp plot ########################
    # extent = [35.362, 36.992, -102.443, -100.00]
    #
    # first_dt = _format_date_time(case_steps.iloc[0]['date'], case_steps.iloc[0]['mrms-time'])
    # last_dt = _format_date_time(case_steps.iloc[-1]['date'], case_steps.iloc[-1]['mrms-time'])
    #
    # viz_files = goes_utils.get_abi_files(local_abi_path, 'goes16', 'ABI-L1b-Rad',
    #       first_dt, last_dt, 'M2', '2', prompt=False)
    #
    # inf_files = goes_utils.get_abi_files(local_abi_path, 'goes16', 'ABI-L2-CMIP',
    #       first_dt, last_dt, 'M2', '13', prompt=False)
    #
    # total_files = len(viz_files)
    #
    # print('\n')
    # for idx, viz_file in enumerate(viz_files):
    #     viz_data = goes_utils.read_file(viz_file)
    #     inf_data = goes_utils.read_file(inf_files[idx])
    #     scan_time = viz_data['scan_date']
    #     print('Processing: {} ({}/{})'.format(scan_time, idx + 1, total_files))
    #     print('GOES vis metadata: {}'.format(viz_data['scan_date']))
    #     print('GOES inf metadata: {}'.format(inf_data['scan_date']))
    #     time = datetime.strftime(scan_time, '%H%M')
    #     date = datetime.strftime(scan_time, '%m%d%Y')
    #
    #
    #     make_wtlma_glm_mercator_dual_hitemp(local_wtlma_path, local_glm_path, date, time,
    #               wwa_fname, (viz_data, inf_data), 2, extent, window=False, show=False,
    #               save=True, outpath=img_outpath)
    #
    #     print('------------------------------------------------------------------------------')
    #     break
    #######################################################################


    ############################# Main loop ###############################
    # Pull the ABI files
    first_dt = _format_date_time(case_steps.iloc[0]['date'], case_steps.iloc[0]['mrms-time'])
    last_dt = _format_date_time(case_steps.iloc[-1]['date'], case_steps.iloc[-1]['mrms-time'])

    #extent = [33.66, 37.7, -103.735, -97.87]
    extent = [35.362, 36.992, -102.443, -100.00]
    #LMA centered extent: extent=None
    extent = [35, 36.5, -102.5, -100]

    viz_files = goes_utils.get_abi_files_dict(local_abi_path, 'goes16', 'ABI-L1b-Rad',
            first_dt, last_dt, 'M2', '2', prompt=False)

    inf_files = goes_utils.get_abi_files_dict(local_abi_path, 'goes16', 'ABI-L2-CMIP',
            first_dt, last_dt, 'M2', '13', prompt=False)

    total_files = len(viz_files)
    print('\n')
    for idx, step in case_steps.iterrows():
        point1 = (step['lat1'], step['lon1'])
        point2 = (step['lat2'], step['lon2'])

        point1 = grib.trunc(point1, 3)
        point2 = grib.trunc(point2, 3)

        step_date = step['date']          # Format: MMDDYYYY
        step_time = step['mrms-time']     # Format: HHMM

        print('Processing: {}-{}z ({}/{})'.format(step_date, step_time, idx + 1, total_files))

        viz_data = goes_utils.read_file(viz_files[step_time])
        inf_data = goes_utils.read_file(inf_files[step_time])

        print('GOES vis metadata: {}'.format(viz_data['scan_date']))
        print('GOES inf metadata: {}'.format(inf_data['scan_date']))

        sat_data = (viz_data, inf_data)
        if (step_time != '2206'): # MISSING FILE
            # make_mrms_glm_plot(local_mrms_path, local_glm_path, local_wtlma_path,
            #       step_date, step_time, point1, point2, memmap_path)

            # make_mrms_xsect2(local_mrms_path, local_wtlma_path, step_date, step_time,
            #         point1, point2, show=False, save=True, outpath=img_outpath)

            make_wtlma_glm_mercator_dual(local_wtlma_path, local_glm_path, step_date,
                  step_time, point1, point2, wwa_fname, sat_data, 2, extent, show=True,
                  save=False, outpath=img_outpath)
        print('------------------------------------------------------------------------------')
        exit(0)
    #######################################################################


    ######################## plot_sammich_mercator ########################
    # first_dt = _format_date_time(case_steps.iloc[0]['date'], case_steps.iloc[0]['mrms-time'])
    # viz_files = goes_utils.get_abi_files(local_abi_path, 'goes16', 'ABI-L1b-Rad',
    #        first_dt, first_dt, 'M2', '2', prompt=False) # first_dt[:-5] + '20:55'
    #
    # inf_files = goes_utils.get_abi_files(local_abi_path, 'goes16', 'ABI-L2-CMIP',
    #       first_dt, first_dt, 'M2', '13', prompt=False)
    # # Maryland --> extent=[36.62, 40.78, -80.36, -72.03]
    # viz_data = goes_utils.read_file(viz_files[0], extent=[33.66, 37.7, -103.735, -97.87])
    # inf_data = goes_utils.read_file(inf_files[0], extent=[33.66, 37.7, -103.735, -97.87])
    # # viz_data = goes_utils.read_file(viz_files[0])
    # # inf_data = goes_utils.read_file(inf_files[0])
    # #goes_utils.plot_sammich_geos(viz_data, inf_data)
    # goes_utils.plot_sammich_mercator(viz_data, inf_data)
    #######################################################################


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
