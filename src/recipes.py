import sys
import six
import pandas as pd
from sys import exit, getrefcount
from datetime import datetime
from os.path import isfile
from os import remove

import wtlma
import goesawsinterface
import grib
import localglminterface
import glm_utils
import plotting_funcs
import plotting_utils
import goes_utils



def driver(paths, case_coords, extent, sat_meta, func_name, plot_sets,
           func_ext=None, func_mode=None, points=None):

    pull_sat = {
        'wtlma_plot': False,
        'mrms_glm_plot': False,
        'mrms_xsect2': False,
        'wtlma_glm_mercator_dual': True,
        'wtlma_glm_mercator_dual_2': True,
        'wtlma_glm_mercator_dual_hitemp': True,
        'plot_merc_abi_mrms': True,
        'plot_mrms_lma_abi_glm': True
    }

    vis_files = None
    inf_files = None

    d_dict = {'date': str, 'wsr-time': str, 'mrms-time': str, 'lat1': float,
              'lon1': float, 'lat2': float, 'lon2': float}
    case_steps = pd.read_csv(case_coords, sep=',', header=0, dtype=d_dict)

    first_dt = _format_date_time(case_steps.iloc[0]['date'], case_steps.iloc[0]['mrms-time'])
    last_dt = _format_date_time(case_steps.iloc[-1]['date'], case_steps.iloc[-1]['mrms-time'])

    if (pull_sat[func_name] == True):
        if (func_ext == 'hitemp'):  # Call get_abi_files()
            vis_files = goes_utils.get_abi_files(paths['local_abi_path'], sat_meta['satellite'],
                                    sat_meta['vis_prod'],first_dt, last_dt, sat_meta['sector'],
                                    sat_meta['vis_chan'], prompt=False)

            inf_files = goes_utils.get_abi_files(paths['local_abi_path'], sat_meta['satellite'],
                                    sat_meta['inf_prod'],first_dt, last_dt, sat_meta['sector'],
                                    sat_meta['inf_chan'], prompt=False)
        else:   # Call get_abi_files_dict()
            vis_files = goes_utils.get_abi_files_dict(paths['local_abi_path'], sat_meta['satellite'],
                                    sat_meta['vis_prod'],first_dt, last_dt, sat_meta['sector'],
                                    sat_meta['vis_chan'], prompt=False)

            inf_files = goes_utils.get_abi_files_dict(paths['local_abi_path'], sat_meta['satellite'],
                                    sat_meta['inf_prod'],first_dt, last_dt, sat_meta['sector'],
                                    sat_meta['inf_chan'], prompt=False)

    if (func_ext == 'hitemp'):
        total_files = len(vis_files)
        print('\n')
        for idx, vis_file in enumerate(vis_files):
            vis_data = goes_utils.read_file(vis_file)
            inf_data = goes_utils.read_file(inf_files[idx])
            scan_time = vis_data['scan_date']

            step_meta = 'Processing: {} ({}/{})'.format(scan_time, idx + 1, total_files)
            geo_extent = 'Geospatial extent: {}'.format(extent)

                                                # Keep extra space after "chan-{}"
            goes_vis_meta = 'Sat vis: {} {} Sec-{} Chan-{}  {}'.format(sat_meta['satellite'],
                        sat_meta['vis_prod'], sat_meta['sector'], sat_meta['vis_chan'],
                        vis_data['scan_date'])

            goes_inf_meta = 'Sat inf: {} {} Sec-{} Chan-{} {}'.format(sat_meta['satellite'],
                        sat_meta['inf_prod'], sat_meta['sector'], sat_meta['inf_chan'],
                        inf_data['scan_date'])

            print(step_meta)
            print(geo_extent)
            print(goes_vis_meta)
            print(goes_inf_meta)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(step_meta + '\n')
                logfile.write(func_name + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            time = datetime.strftime(scan_time, '%H%M')
            date = datetime.strftime(scan_time, '%m%d%Y')

            if (func_name == 'wtlma_glm_mercator_dual'):
                make_wtlma_glm_mercator_dual(
                            paths['local_wtlma_path'],
                            paths['local_glm_path'],
                            date,
                            time,
                            None,
                            None,
                            paths['wwa'],
                            (vis_data, inf_data),
                            func_mode,
                            extent,
                            window=sat_meta['glm_5min'],
                            show=plot_sets['show'],
                            save=plot_sets['save'],
                            outpath=paths['img_outpath'],
                            logpath=paths['logpath']
                            )
            else:
                raise ValueException('Invalid function name')

            fin = '------------------------------------------------------------------------------'
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')

    else:
        print('\n')
        for idx, step in case_steps.iterrows():
            point1 = (step['lat1'], step['lon1'])
            point2 = (step['lat2'], step['lon2'])

            point1 = grib.trunc(point1, 3)
            point2 = grib.trunc(point2, 3)

            step_date = step['date']          # Format: MMDDYYYY
            step_time = step['mrms-time']     # Format: HHMM

            step_meta = 'Processing: {}-{}z ({}/-)'.format(step_date, step_time, idx + 1)

            if (vis_files):
                vis_data = goes_utils.read_file(vis_files[step_time])
                                                            # Keep extra space after "chan-{}"
                goes_vis_meta = 'Sat vis: {} {} Sec-{} Chan-{}  {}'.format(sat_meta['satellite'],
                            sat_meta['vis_prod'], sat_meta['sector'], sat_meta['vis_chan'],
                            vis_data['scan_date'])
            else:
                vis_data = None
                goes_vis_meta = 'No visible satellite data'

            if (inf_files):
                inf_data = goes_utils.read_file(inf_files[step_time])
                goes_inf_meta = 'Sat inf: {} {} Sec-{} Chan-{} {}'.format(sat_meta['satellite'],
                            sat_meta['inf_prod'], sat_meta['sector'], sat_meta['inf_chan'],
                            inf_data['scan_date'])
            else:
                inf_data = None
                goes_inf_meta = 'No infrared satellite data'

            geo_extent = 'Geospatial extent: {}'.format(extent)

            print(step_meta)
            print(geo_extent)
            print(goes_vis_meta)
            print(goes_inf_meta)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(step_meta + '\n')
                logfile.write(func_name + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            sat_data = (vis_data, inf_data)
            if (step_time != '2206'): # MISSING FILE
                if (func_name == 'mrms_glm_plot'):
                    make_mrms_glm_plot(
                                paths['local_mrms_path'],
                                paths['local_glm_path'],
                                paths['local_wtlma_path'],
                                step_date,
                                step_time,
                                point1,
                                point2,
                                paths['memmap_path'],
                                paths['wwa'],
                                show=plot_sets['show'],
                                save=plot_sets['save'],
                                outpath=paths['img_outpath']
                                )

                elif (func_name == 'mrms_xsect2'):
                    make_mrms_xsect2(
                                paths['local_mrms_path'],
                                paths['local_wtlma_path'],
                                step_date,
                                step_time,
                                point1,
                                point2,
                                show=plot_sets['show'],
                                save=plot_sets['save'],
                                outpath=paths['img_outpath']
                                )

                elif (func_name == 'wtlma_glm_mercator_dual'):
                    make_wtlma_glm_mercator_dual(
                                paths['local_wtlma_path'],
                                paths['local_glm_path'],
                                date,
                                time,
                                None,
                                None,
                                paths['wwa'],
                                (vis_data, inf_data),
                                func_mode,
                                extent,
                                window=sat_meta['glm_5min'],
                                show=plot_sets['show'],
                                save=plot_sets['save'],
                                outpath=paths['img_outpath'],
                                logpath=paths['logpath']
                                )
                elif (func_name == 'plot_merc_abi_mrms'):
                    make_merc_abi_mrms(
                                paths['local_mrms_path'],
                                sat_data[1],
                                step_date,
                                step_time,
                                extent,
                                paths['memmap_path'],
                                paths['wwa'],
                                points_to_plot=(point1, point2),
                                show=plot_sets['show'],
                                save=plot_sets['save'],
                                outpath=paths['img_outpath'])
                elif (func_name == 'plot_mrms_lma_abi_glm'):
                    make_mrms_lma_abi_glm(
                                sat_data,
                                paths['local_mrms_path'],
                                paths['local_glm_path'],
                                paths['local_wtlma_path'],
                                step_date,
                                step_time,
                                None,
                                None,
                                extent,
                                paths['memmap_path'],
                                paths['wwa'],
                                show=plot_sets['show'],
                                save=plot_sets['save'],
                                outpath=paths['img_outpath']),
                                logpath=paths['logpath']
                    )
                else:
                    raise ValueError("Invalid function name")
            fin = '------------------------------------------------------------------------------'
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')



def make_mrms_lma_abi_glm(sat_data, local_mrms_path, local_glm_path, local_wtlma_path,
                          date, time, point1, point2, extent, memmap_path, wwa_fname,
                          show=True, save=False, outpath=None, logpath=None):

    ext_point1 = (extent[0], extent[2])
    ext_point2 = (extent[1], extent[3])

    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    grid_extent = {'min_lat': extent[0], 'max_lat': extent[1],
                   'min_lon': extent[2], 'max_lon': extent[3]}

    mrms_obj = plotting_utils.get_composite_ref(local_mrms_path, time, ext_point1,
                            ext_point2, memmap_path)

    glm_scans = localglminterface.get_files_in_range(local_glm_path, dt, dt)
    glm_scans.sort(key=lambda x: x.filename.split('.')[1])
    glm_scan_idx = 0

    glm_meta1 = 'GLM 5-min window: {}'.format(window)
    glm_meta2 = 'GLM Metadata: {} {}z {}'.format(glm_scans[glm_scan_idx].scan_date,
                            glm_scans[glm_scan_idx].scan_time, glm_scans[glm_scan_idx].filename)

    print(glm_meta1)
    print(glm_meta2)

    glm_data = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path, meta=True, window=window)

    lma_files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    lma_fname = lma_files[0]
    lma_abs_path = wtlma._parse_abs_path(local_wtlma_path, lma_fname)
    lma_obj = wtlma.parse_file(lma_abs_path, sub_t=sub_time)

    if (logpath is not None):
        with open(logpath, 'a') as logfile:
            logfile.write(glm_meta1 + '\n')
            logfile.write(glm_meta2 + '\n')
            logfile.write('WTLMA filename: {}\n'.format(lma_fname))
            logfile.write('WTLMA subset time: {}\n'.format(sub_time))

    wwa_polys = plotting_utils.get_wwa_polys(wwa_fname, date, time, wwa_type=['SV', 'TO'])

    plotting_funcs.plot_mrms_lma_abi_glm(sat_data, mrms_obj, glm_obj, lma_obj,
                    grid_extent=grid_extent, points_to_plot=None, range_rings=True,
                    wwa_polys=Nwwa_polys, show=show, save=save, outpath=outpath,
                    logpath=logpath)



def make_merc_abi_mrms(local_mrms_path, sat_data, date, time, extent, memmap_path,
                       wwa_fname, points_to_plot=None, show=True, save=False, outpath=None):
    """
    points_to_plot : list of tuples, optional
        Coordinate pairs to plot
        Format: [(lat1, lon1), (lat2, lon2)]
    """

    ext_point1 = (extent[0], extent[2])
    ext_point2 = (extent[1], extent[3])

    grid_extent = {'min_lat': extent[0], 'max_lat': extent[1],
                   'min_lon': extent[2], 'max_lon': extent[3]}

    mrms_obj = plotting_utils.get_composite_ref(local_mrms_path, time, ext_point1,
                            ext_point2, memmap_path)

    wwa_polys = plotting_utils.get_wwa_polys(wwa_fname, date, time, wwa_type=['SV', 'TO'])

    plotting_funcs.plot_merc_abi_mrms(sat_data, mrms_obj, grid_extent=grid_extent,
                            points_to_plot=points_to_plot, range_rings=False,
                            wwa_polys=wwa_polys, show=show, save=save, outpath=outpath)



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
                       time, point1, point2, memmap_path, wwa_fname, show=True,
                       save=False, outpath=None):
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
    glm_scans.sort(key=lambda x: x.filename.split('.')[1])

    glm_scan_idx = 0 # Selects lowest filename ending
    # ex: IXTR99_KNES_232054_40202.2019052320 <--
    #     IXTR99_KNES_232054_14562.2019052321

    # 2 files for each time, apparently from 2 diff sources but same data
    glm_obj = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path, meta=True)

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
    filtered_data, wtlma_coords = plotting_utils.filter_by_dist(wtlma_data.data, 4000,
                                                                point1, point2, 100)
    wtlma_data._set_data(filtered_data)

    cross_data, lats, lons = plotting_funcs.process_slice(local_mrms_path, time,
                                                          point1, point2)

    plotting_funcs.plot_mrms_cross_section2(data=cross_data, lons=lons, lats=lats,
                                            wtlma_obj=wtlma_data, wtlma_coords=wtlma_coords,
                                            show=show, save=save, outpath=outpath)



def make_wtlma_glm_mercator_dual(local_wtlma_path, local_glm_path, date, time,
         point1, point2, wwa_fname, sat_data, func, plot_extent, window=False,
         show=True, save=False, outpath=None, logpath=None):

    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    # grid_extent = None
    grid_extent = {'min_lat': plot_extent[0], 'max_lat': plot_extent[1],
                   'min_lon': plot_extent[2], 'max_lon': plot_extent[3]}

    glm_scans = localglminterface.get_files_in_range(local_glm_path, dt, dt)
    glm_scans.sort(key=lambda x: x.filename.split('.')[1])

    glm_scan_idx = 0 # Selects lowest filename ending
    # ex: IXTR99_KNES_232054_40202.2019052320 <--
    #     IXTR99_KNES_232054_14562.2019052321

    glm_meta1 = 'GLM 5-min window: {}'.format(window)
    glm_meta2 = 'GLM metadata: {} {}z {}'.format(glm_scans[glm_scan_idx].scan_date,
                glm_scans[glm_scan_idx].scan_time, glm_scans[glm_scan_idx].filename)

    print(glm_meta1)
    print(glm_meta2)
    # 2 files for each time, apparently from 2 diff sources but same data
    glm_data = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path, meta=True, window=window)

    print('Fetching LMA data...')
    files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    wtlma_fname = files[0]
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)

    if (logpath is not None):
        with open(logpath, 'a') as logfile:
            logfile.write(glm_meta1 + '\n')
            logfile.write(glm_meta2 + '\n')
            logfile.write('WTLMA filename: {}\n'.format(wtlma_fname))
            logfile.write('WTLMA subset time: {}\n'.format(sub_time))

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
    elif (func == 3):
        plotting_funcs.plot_merc_glm_lma_sbs(glm_data, wtlma_data, points_to_plot=None,
                                            range_rings=True, wwa_polys=wwa_polys,
                                            satellite_data=sat_data, grid_extent=grid_extent,
                                            show=show, save=save, outpath=outpath)
    else:
        raise ValueError('Invalid func param, must be 1, 2, or 3')



def make_wtlma_glm_mercator_dual_hitemp(local_wtlma_path, local_glm_path, date, time,
         wwa_fname, sat_data, func, plot_extent, window=False, show=True, save=False,
         outpath=None, logpath=None):

    dt = _format_date_time(date, time)
    sub_time = _format_time_wtlma(time)

    # grid_extent = None
    grid_extent = {'min_lat': plot_extent[0], 'max_lat': plot_extent[1],
                   'min_lon': plot_extent[2], 'max_lon': plot_extent[3]}

    glm_scans = localglminterface.get_files_in_range(local_glm_path, dt, dt)
    glm_scans.sort(key=lambda x: x.filename.split('.')[1])

    glm_scan_idx = 0 # Selects lowest filename ending
    # ex: IXTR99_KNES_232054_40202.2019052320 <--
    #     IXTR99_KNES_232054_14562.2019052321

    glm_meta1 = 'GLM 5-min window: {}'.format(window)
    glm_meta2 = 'GLM metadata: {} {}z {}'.format(glm_scans[glm_scan_idx].scan_date,
                glm_scans[glm_scan_idx].scan_time, glm_scans[glm_scan_idx].filename)

    print(glm_meta1)
    print(glm_meta2)

    # 2 files for each time, apparently from 2 diff sources but same data
    glm_data = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path, meta=True, window=window)

    print('Fetching LMA data...')
    files = wtlma.get_files_in_range(local_wtlma_path, dt, dt)
    wtlma_fname = files[0]
    wtlma_abs_path = wtlma._parse_abs_path(local_wtlma_path, files[0])
    wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)

    if (logpath is not None):
        with open(logpath, 'a') as logfile:
            logfile.write(glm_meta1 + '\n')
            logfile.write(glm_meta2 + '\n')
            logfile.write('WTLMA filename: {}\n'.format(wtlma_fname))
            logfile.write('WTLMA subset time: {}\n'.format(sub_time))

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
    elif (func == 3):
        plotting_funcs.plot_merc_glm_lma_sbs(glm_data, wtlma_data, points_to_plot=None,
                                            range_rings=True, wwa_polys=wwa_polys,
                                            satellite_data=sat_data, grid_extent=grid_extent,
                                            show=show, save=save, outpath=outpath)
    else:
        raise ValueError('Invalid func param, must be 1, 2, or 3')



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
