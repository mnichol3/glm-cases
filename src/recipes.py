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



def read_case_steps(f_path):
    """
    Reads the file containing the date & times for a specified case

    Parameters
    ----------
    f_path : str
        Absolute path of the text file that contains the case metadata

    Returns
    -------
    case_steps : DataFrame
        DataFrame where each row is a time step in the case
    """
    d_dict = {'date': str, 'wsr-time': str, 'mrms-time': str, 'lat1': float,
              'lon1': float, 'lat2': float, 'lon2': float}
    case_steps = pd.read_csv(f_path, sep=',', header=0, dtype=d_dict)

    return case_steps



def get_datetime_bookends(case_df):
    """
    Gets the first & last date and time for a case

    Parameters
    ----------
    case_df : DataFrame
        DataFrame containing case metadata

    Returns
    -------
    Tuple of str
        The first date & time and the last date & time of the case
        Format: MM-DD-YYYY-HH:MM
    """
    first_t1 = _format_date_time(case_df.iloc[0]['date'], case_df.iloc[0]['mrms-time'])
    last_t1 = _format_date_time(case_df.iloc[-1]['date'], case_df.iloc[-1]['mrms-time'])

    return (first_t1, last_t1)



def get_sat_data(first_t1, last_t1, sat_meta, paths, vis=True, inf=True, file_dict=False):
    """
    Retrieves the satellite imagery for the period defined by first_t1 & last_t1,
    inclusive

    Parameters
    ----------
    first_t1 : str
        First date & time string that defines the period
        Format: MM-DD-YYYY-HH:MM
    last_t1 : str
        Ending date & time string that defines the period
        Format: MM-DD-YYYY-HH:MM
    sat_meta : dict; key: str, val: str
        Dictionary containing metadata about the satellite imagery to retrieve.
        Keys: satellite, vis_prod, inf_prod, sector, vis_chan, inf_chan, glm_5min
    vis : bool, optional
        Determines whether or not to retrieve the visible data defined in sat_meta.
        Default = True
    inf : bool, optional
        Determines whether or not to retrieve the infrared data defined in sat_meta.
        Default = True
    file_dict : bool, optional
        If True, satellite imagery filenamess are returned as a dictionary, where
        the key is the scan time (str) and the value is the filename (str). IF False,
        the imagery filenames are returned as a list of str
        Default = False

    Returns
    -------
    Tuple of either list or dict
        Tuple containing the ABI file names
        Format: (visible, infrared)
    """
    vis_files = None
    inf_files = None

    if (vis):
        if (file_dict):
            vis_files = goes_utils.get_abi_files_dict(paths['local_abi_path'],
                            sat_meta['satellite'], sat_meta['vis_prod'], first_t1,
                            last_t1, sat_meta['sector'], sat_meta['vis_chan'],
                            prompt=False)
        else:
            vis_files = goes_utils.get_abi_files(paths['local_abi_path'], sat_meta['satellite'],
                            sat_meta['vis_prod'], first_t1, last_t1, sat_meta['sector'],
                            sat_meta['vis_chan'], prompt=False)
    if (inf):
        if (file_dict):
            inf_files = goes_utils.get_abi_files_dict(paths['local_abi_path'],
                            sat_meta['satellite'], sat_meta['inf_prod'], first_t1,
                            last_t1, sat_meta['sector'], sat_meta['inf_chan'],
                            prompt=False)
        else:
            inf_files = goes_utils.get_abi_files(paths['local_abi_path'], sat_meta['satellite'],
                            sat_meta['inf_prod'], first_t1, last_t1, sat_meta['sector'],
                            sat_meta['inf_chan'], prompt=False)

    return (vis_files, inf_files)



def make_mrms_lma_abi_glm(paths, sat_meta, plot_set, extent, hitemp=True):
    """
    Gathers all the data to call plot_mrms_lma_abi_glm()

    Parameters
    ----------
    paths : dict; key: str, val: str
        Dict containing local directory paths for the various data sources
    sat_meta : dict; key: str, val: str
        Dict containing metadata about the satellite imagery to retrieve.
        Keys: satellite, vis_prod, inf_prod, sector, vis_chan, inf_chan, glm_5min
    plot_set : dict; key: str, val: str
        Dict containing booleans dictating showing & saving the plots
        Keys: save, show
    extent : list of floats
        Geographic extent of the plot.
        Format: [min_lat, max_lat, min_lon, max_lon]
    hitemp : bool, optional
        If True, imagery is created at 1-min timesteps. If False, imagery is
        created according to MRMS timesteps
    """

    point1 = None
    point2 = None

    case_steps = read_case_steps(paths['case_coords'])
    first_t1, last_t1 = get_datetime_bookends(case_steps)

    grid_extent = {'min_lat': extent[0], 'max_lat': extent[1],
                   'min_lon': extent[2], 'max_lon': extent[3]}

    ext_point1 = (extent[0], extent[2])
    ext_point2 = (extent[1], extent[3])

    if (hitemp):
        vis_files, inf_files = get_sat_data(first_t1, last_t1, sat_meta, paths,
                                vis=True, inf=True, file_dict=False)

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
                logfile.write('make_mrms_lma_abi_glm-hitemp\n')
                logfile.write(step_meta + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            time = datetime.strftime(scan_time, '%H%M')
            date = datetime.strftime(scan_time, '%m%d%Y')

            t1 = _format_date_time(date, time)
            sub_time = _format_time_wtlma(time)

            # Only get new MRMS object if new mrms-time is reached
            if (time in list(case_steps['mrms-time'])):
                mrms_obj = plotting_utils.get_composite_ref(paths['local_mrms_path'],
                                            time, ext_point1, ext_point2, paths['memmap_path'])
                df_row = case_steps.loc[case_steps['mrms-time'] == time]
                point1 = (df_row.iloc[0]['lat1'], df_row.iloc[0]['lon1'])
                point2 = (df_row.iloc[0]['lat2'], df_row.iloc[0]['lon2'])

            ### Get GLM data ###
            glm_scans = localglminterface.get_files_in_range(paths['local_glm_path'],
                                                             t1, t1)
            glm_scans.sort(key=lambda x: x.filename.split('.')[1])
            glm_scan_idx = 0

            glm_meta1 = 'GLM 5-min window: {}'.format(sat_meta['glm_5min'])
            glm_meta2 = 'GLM Metadata: {} {}z {}'.format(
                                    glm_scans[glm_scan_idx].scan_date,
                                    glm_scans[glm_scan_idx].scan_time,
                                    glm_scans[glm_scan_idx].filename
                                    )

            print(glm_meta1)
            print(glm_meta2)

            glm_obj = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path,
                                    meta=True, window=sat_meta['glm_5min'])

            lma_files = wtlma.get_files_in_range(paths['local_wtlma_path'], t1, t1)
            lma_fname = lma_files[0]
            lma_abs_path = wtlma._parse_abs_path(paths['local_wtlma_path'], lma_fname)
            lma_obj = wtlma.parse_file(lma_abs_path, sub_t=sub_time)

            if (paths['logpath'] is not None):
                with open(paths['logpath'], 'a') as logfile:
                    logfile.write(glm_meta1 + '\n')
                    logfile.write(glm_meta2 + '\n')
                    logfile.write('WTLMA filename: {}\n'.format(lma_fname))
                    logfile.write('WTLMA subset time: {}\n'.format(sub_time))

            wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                                     wwa_type=['SV', 'TO'])

            if (point1 == None or point2 == None):
                points_to_plot = None
            else:
                points_to_plot = (point1, point2)

            plotting_funcs.plot_mrms_lma_abi_glm((vis_data, inf_data), mrms_obj,
                            glm_obj, lma_obj, grid_extent=grid_extent, points_to_plot=points_to_plot,
                            range_rings=True, wwa_polys=wwa_polys, show=plot_set['show'],
                            save=plot_set['save'], outpath=paths['outpath'])

            fin = ('---------------------------------------'
                   '---------------------------------------')
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')

    else: # Not high temporal res - go by mrms file times
        vis_files, inf_files = get_sat_data(first_t1, last_t1, sat_meta,
                                    paths, vis=True, inf=True, file_dict=True)

        print('\n')
        for idx, step in case_steps.iterrows():
            date = step['date']
            time = step['mrms-time']

            point1 = (step['lat1'], step['lon1'])
            point2 = (step['lat2'], step['lon2'])

            point1 = grib.trunc(point1, 3)
            point2 = grib.trunc(point2, 3)

            step_meta = 'Processing: {}-{}z ({}/-)'.format(date, time, idx + 1)

            geo_extent = 'Geospatial extent: {}'.format(extent)

            vis_data = goes_utils.read_file(vis_files[step['mrms-time']])
            inf_data = goes_utils.read_file(inf_files[step['mrms-time']])

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
                logfile.write('make_mrms_lma_abi_glm\n')
                logfile.write(step_meta + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            ### Get MRMS Composite Ref ###
            mrms_obj = plotting_utils.get_composite_ref(paths['local_mrms_path'],
                                        time, ext_point1, ext_point2, paths['memmap_path'])

            t1 = _format_date_time(date, time)
            sub_time = _format_time_wtlma(time)

            ### Get GLM data ###
            glm_scans = localglminterface.get_files_in_range(paths['local_glm_path'],
                                                             t1, t1)
            glm_scans.sort(key=lambda x: x.filename.split('.')[1])
            glm_scan_idx = 0

            glm_meta1 = 'GLM 5-min window: {}'.format(sat_meta['glm_5min'])
            glm_meta2 = 'GLM Metadata: {} {}z {}'.format(
                                    glm_scans[glm_scan_idx].scan_date,
                                    glm_scans[glm_scan_idx].scan_time,
                                    glm_scans[glm_scan_idx].filename
                                    )

            print(glm_meta1)
            print(glm_meta2)

            glm_obj = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path,
                                    meta=True, window=sat_meta['glm_5min'])

            ### GET WTLMA Data ###
            lma_files = wtlma.get_files_in_range(paths['local_wtlma_path'], t1, t1)
            lma_fname = lma_files[0]
            lma_abs_path = wtlma._parse_abs_path(paths['local_wtlma_path'], lma_fname)
            lma_obj = wtlma.parse_file(lma_abs_path, sub_t=sub_time)

            if (paths['logpath'] is not None):
                with open(paths['logpath'], 'a') as logfile:
                    logfile.write(glm_meta1 + '\n')
                    logfile.write(glm_meta2 + '\n')
                    logfile.write('WTLMA filename: {}\n'.format(lma_fname))
                    logfile.write('WTLMA subset time: {}\n'.format(sub_time))

            wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                            wwa_type=['SV', 'TO'])

            plotting_funcs.plot_mrms_lma_abi_glm((vis_data, inf_data), mrms_obj,
                            glm_obj, lma_obj, grid_extent=grid_extent, points_to_plot=(point1, point2),
                            range_rings=True, wwa_polys=wwa_polys, show=plot_set['show'],
                            save=plot_set['save'], outpath=paths['outpath'])

            fin = ('---------------------------------------'
                   '---------------------------------------')
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')



def make_merc_abi_mrms(paths, sat_meta, plot_set, extent, hitemp=False):
    """
    Gathers all the data to call plot_merc_abi_mrms()

    Parameters
    ----------
    paths : dict; key: str, val: str
        Dict containing local directory paths for the various data sources
    sat_meta : dict; key: str, val: str
        Dict containing metadata about the satellite imagery to retrieve.
        Keys: satellite, vis_prod, inf_prod, sector, vis_chan, inf_chan, glm_5min
    plot_set : dict; key: str, val: str
        Dict containing booleans dictating showing & saving the plots
        Keys: save, show
    extent : list of floats
        Geographic extent of the plot.
        Format: [min_lat, max_lat, min_lon, max_lon]
    hitemp : bool, optional
        If True, imagery is created at 1-min timesteps. If False, imagery is
        created according to MRMS timesteps
    """

    point1 = None
    point2 = None

    case_steps = read_case_steps(paths['case_coords'])
    first_t1, last_t1 = get_datetime_bookends(case_steps)

    grid_extent = {'min_lat': extent[0], 'max_lat': extent[1],
                   'min_lon': extent[2], 'max_lon': extent[3]}

    ext_point1 = (extent[0], extent[2])
    ext_point2 = (extent[1], extent[3])

    if (hitemp):
        vis_files, inf_files = get_sat_data(first_t1, last_t1, sat_meta, paths,
                                vis=True, inf=True, file_dict=False)

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
                logfile.write('make_merc_abi_mrms-hitemp\n')
                logfile.write(step_meta + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            time = datetime.strftime(scan_time, '%H%M')
            date = datetime.strftime(scan_time, '%m%d%Y')

            # Only get new MRMS object if new mrms-time is reached
            if (time in list(case_steps['mrms-time'])):
                mrms_obj = plotting_utils.get_composite_ref(paths['local_mrms_path'],
                                            time, ext_point1, ext_point2, paths['memmap_path'])
                df_row = case_steps.loc[case_steps['mrms-time'] == time]
                point1 = (df_row.iloc[0]['lat1'], df_row.iloc[0]['lon1'])
                point2 = (df_row.iloc[0]['lat2'], df_row.iloc[0]['lon2'])

            wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                                     wwa_type=['SV', 'TO'])

            if (point1 == None or point2 == None):
                points_to_plot = None
            else:
                points_to_plot = (point1, point2)

            plotting_funcs.plot_merc_abi_mrms(sat_data, mrms_obj, grid_extent=grid_extent,
                                    points_to_plot=points_to_plot, range_rings=True,
                                    wwa_polys=wwa_polys, show=plot_set['show'],
                                    save=plot_set['save'], outpath=paths['outpath'])

            fin = ('---------------------------------------'
                   '---------------------------------------')
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')

    else: # Not high temporal res - go by mrms file times
        vis_files, inf_files = get_sat_data(first_t1, last_t1, sat_meta,
                                    paths, vis=True, inf=True, file_dict=True)

        print('\n')
        for idx, step in case_steps.iterrows():
            date = step['date']
            time = step['mrms-time']

            point1 = (step['lat1'], step['lon1'])
            point2 = (step['lat2'], step['lon2'])

            point1 = grib.trunc(point1, 3)
            point2 = grib.trunc(point2, 3)

            step_meta = 'Processing: {}-{}z ({}/-)'.format(date, time, idx + 1)

            geo_extent = 'Geospatial extent: {}'.format(extent)

            vis_data = goes_utils.read_file(vis_files[step['mrms-time']])
            inf_data = goes_utils.read_file(inf_files[step['mrms-time']])

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
                logfile.write('make_merc_abi_mrms\n')
                logfile.write(step_meta + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            ### Get MRMS Composite Ref ###
            mrms_obj = plotting_utils.get_composite_ref(paths['local_mrms_path'],
                                        time, ext_point1, ext_point2, paths['memmap_path'])

            wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                            wwa_type=['SV', 'TO'])

            plotting_funcs.plot_merc_abi_mrms(sat_data, mrms_obj, grid_extent=grid_extent,
                                    points_to_plot=points_to_plot, range_rings=True,
                                    wwa_polys=wwa_polys, show=plot_set['show'],
                                    save=plot_set['save'], outpath=paths['outpath'])

            fin = ('---------------------------------------'
                   '---------------------------------------')
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')



def make_wtlma_plot(paths, start, stop, points_to_plot=None):
    """
    Depricated dont use
    Gathers all the data to call plot_wtlma()

    Parameters
    ----------
    paths : dict; key: str, val: str
        Dict containing local directory paths for the various data sources
    start : something
    end : also something
    points_to_plot : tuple of things
    """
    # '05-23-2019-21:00', '05-23-2019-21:55'
    lma_path = paths['local_wtlma_path']
    lma_objs = []
    wtlma_files = wtlma.get_files_in_range(lma_path, start, stop)

    for f in wtlma_files:
        abs_path = wtlma._parse_abs_path(lma_path, f)
        lma_objs.append(wtlma.parse_file(abs_path))
    plotting_funcs.plot_wtlma(lma_objs, points_to_plot=points_to_plot)



def make_mrms_glm_plot(paths, extent, plot_lma=False):
    """
    Gathers all the data to call plot_mrms_glm()

    Parameters
    ----------
    paths : dict; key: str, val: str
        Dict containing local directory paths for the various data sources
    sat_meta : dict; key: str, val: str
        Dict containing metadata about the satellite imagery to retrieve.
        Keys: satellite, vis_prod, inf_prod, sector, vis_chan, inf_chan, glm_5min
    plot_set : dict; key: str, val: str
        Dict containing booleans dictating showing & saving the plots
        Keys: save, show
    extent : list of floats
        Geographic extent of the plot.
        Format: [min_lat, max_lat, min_lon, max_lon]
    plot_lma : bool, optional
        If True, WTLMA data is plotted
    """
    case_steps = read_case_steps(paths['case_coords'])
    first_t1, last_t1 = get_datetime_bookends(case_steps)

    grid_extent = {'min_lat': extent[0], 'max_lat': extent[1],
                   'min_lon': extent[2], 'max_lon': extent[3]}

    lma_fname = 'None'
    sub_time = 'None'

    ext_point1 = (extent[0], extent[2])
    ext_point2 = (extent[1], extent[3])

    for idx, step in case_steps.iterrows():
        date = step['date']
        time = step['mrms-time']

        point1 = (step['lat1'], step['lon1'])
        point2 = (step['lat2'], step['lon2'])

        point1 = grib.trunc(point1, 3)
        point2 = grib.trunc(point2, 3)

        step_meta = 'Processing: {}-{}z ({}/-)'.format(date, time, idx + 1)

        geo_extent = 'Geospatial extent: {}'.format(extent)

        vis_data = goes_utils.read_file(vis_files[step['mrms-time']])
        inf_data = goes_utils.read_file(inf_files[step['mrms-time']])

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
            logfile.write('make_mrms_glm_plot\n')
            logfile.write(step_meta + '\n')
            logfile.write(geo_extent + '\n')
            logfile.write(goes_vis_meta + '\n')
            logfile.write(goes_inf_meta + '\n')

        mrms_obj = plotting_utils.get_composite_ref(paths['local_mrms_path'],
                                    time, ext_point1, ext_point2, paths['memmap_path'])

        glm_scans = localglminterface.get_files_in_range(paths['local_glm_path'],
                                                         t1, t1)

        glm_scans.sort(key=lambda x: x.filename.split('.')[1])
        glm_scan_idx = 0

        glm_meta1 = 'GLM 5-min window: {}'.format(sat_meta['glm_5min'])
        glm_meta2 = 'GLM Metadata: {} {}z {}'.format(
                                glm_scans[glm_scan_idx].scan_date,
                                glm_scans[glm_scan_idx].scan_time,
                                glm_scans[glm_scan_idx].filename
                                )

        print(glm_meta1)
        print(glm_meta2)

        glm_obj = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path,
                                        meta=True, window=sat_meta['glm_5min'])

        if (plot_lma):
            t1 = _format_date_time(date, time)
            sub_time = _format_time_wtlma(time)

            lma_files = wtlma.get_files_in_range(paths['local_wtlma_path'], t1, t1)
            lma_fname = lma_files[0]
            lma_abs_path = wtlma._parse_abs_path(paths['local_wtlma_path'], lma_fname)
            lma_obj = wtlma.parse_file(lma_abs_path, sub_t=sub_time)

        if (paths['logpath'] is not None):
            with open(paths['logpath'], 'a') as logfile:
                logfile.write(glm_meta1 + '\n')
                logfile.write(glm_meta2 + '\n')
                logfile.write('WTLMA filename: {}\n'.format(lma_fname))
                logfile.write('WTLMA subset time: {}\n'.format(sub_time))

        wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                        wwa_type=['SV', 'TO'])


        plotting_funcs.plot_mrms_glm(mrms_obj, glm_obj, wtlma_obj=lma_obj,
                points_to_plot=[point1, point2], wwa_polys=wwa_polys)

        fin = ('---------------------------------------'
               '---------------------------------------')
        print(fin)

        with open(paths['logpath'], 'a') as logfile:
            logfile.write(fin + '\n')



def make_mrms_xsect2(paths, plot_set, plot_lma=True):
    """
    Contains helper-function calls needed for run_mrms_xsect2()

    Parameters
    ----------
    paths : dict; key: str, val: str
        Dict containing local directory paths for the various data sources
    plot_set : dict; key: str, val: str
        Dict containing booleans dictating showing & saving the plots
        Keys: save, show
    plot_lma : bool, optional
        If True, WTLMA data is plotted
    """
    wtlma_data = None
    wtlma_coords = None

    case_steps = read_case_steps(paths['case_coords'])

    for idx, step in case_steps.iterrows():
        date = step['date']
        time = step['mrms-time']

        point1 = (step['lat1'], step['lon1'])
        point2 = (step['lat2'], step['lon2'])

        point1 = grib.trunc(point1, 3)
        point2 = grib.trunc(point2, 3)

        step_meta = 'Processing: {}-{}z ({}/-)'.format(date, time, idx + 1)

        print(step_meta)

        with open(paths['logpath'], 'a') as logfile:
            logfile.write('make_merc_abi_mrms\n')
            logfile.write(step_meta + '\n')
            logfile.write('X-sect bounds: {}, {}\n'.format(point1, point2))

        if (plot_lma):
            t1 = _format_date_time(date, time)
            sub_time = _format_time_wtlma(time)

            files = wtlma.get_files_in_range(paths['local_wtlma_path'], t1, t1)
            wtlma_abs_path = wtlma._parse_abs_path(paths['local_wtlma_path'], files[0])
            wtlma_data = wtlma.parse_file(wtlma_abs_path, sub_t=sub_time)
            # filter_by_dist(lma_df, dist, start_point, end_point, num_pts) dist in m
            filtered_data, wtlma_coords = plotting_utils.filter_by_dist(wtlma_data.data,
                                                                        4000,
                                                                        point1,
                                                                        point2,
                                                                        100
                                                                        )
            wtlma_data._set_data(filtered_data)

        try:
            cross_data, lats, lons = plotting_funcs.process_slice(paths['local_mrms_path'],
                                                                  time, point1, point2)
        except IndexError as err:
            print('IndexError: {}\n'.format(err))
            print('MRMS time: {}z\n'.format(time))

            with open(paths['logpath'], 'a') as logfile:
                logfile.write('{}\n'.format(err))

            break
        else:
            plotting_funcs.plot_mrms_cross_section2(data=cross_data, lons=lons,
                                                    lats=lats, wtlma_obj=wtlma_data,
                                                    wtlma_coords=wtlma_coords,
                                                    show=plot_set['show'],
                                                    save=plot_set['save'],
                                                    outpath=paths['outpath'])



def make_wtlma_glm_mercator_dual(paths, sat_meta, plot_set, extent, func_num, hitemp=True):
    """
    Gathers all the data to call the plot_mercator_dual family of funcs

    Parameters
    ----------
    paths : dict; key: str, val: str
        Dict containing local directory paths for the various data sources
    sat_meta : dict; key: str, val: str
        Dict containing metadata about the satellite imagery to retrieve.
        Keys: satellite, vis_prod, inf_prod, sector, vis_chan, inf_chan, glm_5min
    plot_set : dict; key: str, val: str
        Dict containing booleans dictating showing & saving the plots
        Keys: save, show
    extent : list of floats
        Geographic extent of the plot.
        Format: [min_lat, max_lat, min_lon, max_lon]
    func_num : int
        Determines which plot_mercator_dual function to ultimately call
        1 --> plot_mercator_dual()
        2 --> plot_mercator_dual_2()
        3 --> plot_mercator_dual_sbs()
    hitemp : bool, optional
        If True, imagery is created at 1-min timesteps. If False, imagery is
        created according to MRMS timesteps
    """
    point1 = None
    point2 = None

    case_steps = read_case_steps(paths['case_coords'])
    first_t1, last_t1 = get_datetime_bookends(case_steps)

    grid_extent = {'min_lat': extent[0], 'max_lat': extent[1],
                   'min_lon': extent[2], 'max_lon': extent[3]}

    ext_point1 = (extent[0], extent[2])
    ext_point2 = (extent[1], extent[3])

    if (hitemp):
        vis_files, inf_files = get_sat_data(first_t1, last_t1, sat_meta, paths,
                                vis=True, inf=True, file_dict=False)

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
                logfile.write('wtlma_glm_mercator_dual-{}-hitemp\n'.format(func_num))
                logfile.write(step_meta + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            time = datetime.strftime(scan_time, '%H%M')
            date = datetime.strftime(scan_time, '%m%d%Y')

            t1 = _format_date_time(date, time)
            sub_time = _format_time_wtlma(time)

            # Only get new MRMS object if new mrms-time is reached
            if (time in list(case_steps['mrms-time'])):
                df_row = case_steps.loc[case_steps['mrms-time'] == time]
                point1 = (df_row.iloc[0]['lat1'], df_row.iloc[0]['lon1'])
                point2 = (df_row.iloc[0]['lat2'], df_row.iloc[0]['lon2'])

            ### Get GLM data ###
            glm_scans = localglminterface.get_files_in_range(paths['local_glm_path'],
                                                             t1, t1)
            glm_scans.sort(key=lambda x: x.filename.split('.')[1])
            glm_scan_idx = 0

            glm_meta1 = 'GLM 5-min window: {}'.format(sat_meta['glm_5min'])
            glm_meta2 = 'GLM Metadata: {} {}z {}'.format(
                                    glm_scans[glm_scan_idx].scan_date,
                                    glm_scans[glm_scan_idx].scan_time,
                                    glm_scans[glm_scan_idx].filename
                                    )

            print(glm_meta1)
            print(glm_meta2)

            glm_data = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path,
                                    meta=True, window=sat_meta['glm_5min'])

            lma_files = wtlma.get_files_in_range(paths['local_wtlma_path'], t1, t1)
            lma_fname = lma_files[0]
            lma_abs_path = wtlma._parse_abs_path(paths['local_wtlma_path'], lma_fname)
            lma_obj = wtlma.parse_file(lma_abs_path, sub_t=sub_time)
            print(lma_obj)

            if (paths['logpath'] is not None):
                with open(paths['logpath'], 'a') as logfile:
                    logfile.write(glm_meta1 + '\n')
                    logfile.write(glm_meta2 + '\n')
                    logfile.write('WTLMA filename: {}\n'.format(lma_fname))
                    logfile.write('WTLMA subset time: {}\n'.format(sub_time))

            wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                                     wwa_type=['SV', 'TO'])

            if (point1 == None or point2 == None):
                points_to_plot = None
            else:
                points_to_plot = (point1, point2)

            if (func_num == 1):
                plotting_funcs.plot_mercator_dual(glm_data, lma_obj,
                                                  points_to_plot=points_to_plot,
                                                  range_rings=True, wwa_polys=wwa_polys,
                                                  satellite_data=(vis_data, inf_data),
                                                  grid_extent=grid_extent,
                                                  show=plot_set['show'],
                                                  save=plot_set['save'],
                                                  outpath=paths['outpath'])
            elif (func_num == 2):
                plotting_funcs.plot_mercator_dual_2(glm_data, lma_obj,
                                                    points_to_plot=points_to_plot,
                                                    range_rings=True, wwa_polys=wwa_polys,
                                                    satellite_data=(vis_data, inf_data),
                                                    grid_extent=grid_extent,
                                                    show=plot_set['show'],
                                                    save=plot_set['save'],
                                                    outpath=paths['outpath'])
            elif (func_num == 3):
                plotting_funcs.plot_merc_glm_lma_sbs(glm_data, lma_obj,
                                                  points_to_plot=points_to_plot,
                                                  range_rings=True, wwa_polys=wwa_polys,
                                                  satellite_data=(vis_data, inf_data),
                                                  grid_extent=grid_extent,
                                                  show=plot_set['show'],
                                                  save=plot_set['save'],
                                                  outpath=paths['outpath'])
            else:
                raise ValueError('Invalid func_num, must be 1, 2, or 3')

            fin = ('---------------------------------------'
                   '---------------------------------------')
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')

    else: # Not high temporal res - go by mrms file times
        vis_files, inf_files = get_sat_data(first_t1, last_t1, sat_meta,
                                    paths, vis=True, inf=True, file_dict=True)

        print('\n')
        for idx, step in case_steps.iterrows():
            date = step['date']
            time = step['mrms-time']

            point1 = (step['lat1'], step['lon1'])
            point2 = (step['lat2'], step['lon2'])

            point1 = grib.trunc(point1, 3)
            point2 = grib.trunc(point2, 3)

            step_meta = 'Processing: {}-{}z ({}/-)'.format(date, time, idx + 1)

            geo_extent = 'Geospatial extent: {}'.format(extent)

            vis_data = goes_utils.read_file(vis_files[step['mrms-time']])
            inf_data = goes_utils.read_file(inf_files[step['mrms-time']])

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
                logfile.write('wtlma_glm_mercator_dual-{}\n'.format(func_num))
                logfile.write(step_meta + '\n')
                logfile.write(geo_extent + '\n')
                logfile.write(goes_vis_meta + '\n')
                logfile.write(goes_inf_meta + '\n')

            t1 = _format_date_time(date, time)
            sub_time = _format_time_wtlma(time)

            ### Get GLM data ###
            glm_scans = localglminterface.get_files_in_range(paths['local_glm_path'],
                                                             t1, t1)
            glm_scans.sort(key=lambda x: x.filename.split('.')[1])
            glm_scan_idx = 0

            glm_meta1 = 'GLM 5-min window: {}'.format(sat_meta['glm_5min'])
            glm_meta2 = 'GLM Metadata: {} {}z {}'.format(
                                    glm_scans[glm_scan_idx].scan_date,
                                    glm_scans[glm_scan_idx].scan_time,
                                    glm_scans[glm_scan_idx].filename
                                    )

            print(glm_meta1)
            print(glm_meta2)

            glm_data = glm_utils.read_file(glm_scans[glm_scan_idx].abs_path,
                                    meta=True, window=sat_meta['glm_5min'])

            ### GET WTLMA Data ###
            lma_files = wtlma.get_files_in_range(paths['local_wtlma_path'], t1, t1)
            lma_fname = lma_files[0]
            lma_abs_path = wtlma._parse_abs_path(paths['local_wtlma_path'], lma_fname)
            lma_obj = wtlma.parse_file(lma_abs_path, sub_t=sub_time)

            if (paths['logpath'] is not None):
                with open(paths['logpath'], 'a') as logfile:
                    logfile.write(glm_meta1 + '\n')
                    logfile.write(glm_meta2 + '\n')
                    logfile.write('WTLMA filename: {}\n'.format(lma_fname))
                    logfile.write('WTLMA subset time: {}\n'.format(sub_time))

            wwa_polys = plotting_utils.get_wwa_polys(paths['wwa_fname'], date, time,
                                            wwa_type=['SV', 'TO'])

            if (func_num == 1):
                plotting_funcs.plot_mercator_dual(glm_data, lma_obj,
                                                  points_to_plot=(point1, point2),
                                                  range_rings=True, wwa_polys=wwa_polys,
                                                  satellite_data=(vis_data, inf_data),
                                                  grid_extent=grid_extent,
                                                  show=plot_set['show'],
                                                  save=plot_set['save'],
                                                  outpath=paths['outpath'])
            elif (func_num == 2):
                plotting_funcs.plot_mercator_dual_2(glm_data, lma_obj,
                                                    points_to_plot=(point1, point2),
                                                    range_rings=True, wwa_polys=wwa_polys,
                                                    satellite_data=(vis_data, inf_data),
                                                    grid_extent=grid_extent,
                                                    show=plot_set['show'],
                                                    save=plot_set['save'],
                                                    outpath=paths['outpath'])
            elif (func_num == 3):
                plotting_funcs.plot_merc_glm_lma_sbs(glm_data, lma_obj,
                                                  points_to_plot=(point1, point2),
                                                  range_rings=True, wwa_polys=wwa_polys,
                                                  satellite_data=(vis_data, inf_data),
                                                  grid_extent=grid_extent,
                                                  show=plot_set['show'],
                                                  save=plot_set['save'],
                                                  outpath=paths['outpath'])
            else:
                raise ValueError('Invalid func_num, must be 1, 2, or 3')


            fin = ('---------------------------------------'
                   '---------------------------------------')
            print(fin)

            with open(paths['logpath'], 'a') as logfile:
                logfile.write(fin + '\n')



def _format_date_time(date, time):
    """
    Helper func to format date & time strings from the case coords file

    Parameters
    ----------
    date : str
        Format: MMDDYYYY
    time: str
        Format: HHMM

    Returns
    -------
    str
        Format: MM-DD-YYYY-HH:MM
    """
    month = date[:2]
    day = date[2:4]
    year = date[-4:]

    hour = time[:2]
    mint = time[2:]
    return '{}-{}-{}-{}:{}'.format(month, day, year, hour, mint)



def _format_time_wtlma(time):
    """
    Formats a time string for WTLMA functions

    Parameters
    ----------
    time : str
        Format: HHMM

    Returns
    -------
    str
        Format: HH:MM
    """
    hr = time[:2]
    mn = time[2:]
    return '{}:{}'.format(hr, mn)
