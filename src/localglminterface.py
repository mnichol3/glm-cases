"""
Author: Matt Nicholson

Functions to handle LocalGLMFile objects
"""
from localglmfile import LocalGLMFile

from os import listdir
from os.path import isfile, join
import re
import sys
import numpy as np
from netCDF4 import Dataset
from datetime import timedelta, datetime


def print_files(base_path, hour=None):
    files = [LocalGLMFile(join(base_path, f)) for f in listdir(base_path) if isfile(join(base_path, f))]

    if (hour is not None):
        filtered = [f for f in files if f.scan_time[0:2] == hour]
        filtered.sort(key=lambda x: x.filename)

        for file in filtered:
            print(file)
    else:
        files.sort(key=lambda x: x.filename)
        for f in files:
            print(f)



def get_files(base_path, hour=None):
    files = [LocalGLMFile(f) for f in listdir(base_path) if isfile(join(base_path, f))]

    if (hour is not None):
        filtered = [f for f in files if f.scan_time[0:2] == hour]
        filtered.sort(key=lambda x: x.filename)

        return filtered
    else:

        files.sort(key=lambda x: x.filename)
        return files



def get_files_in_range(base_path, start, end):
    scans = []
    added = []
    start_dt = datetime.strptime(start, '%m-%d-%Y-%H:%M')
    end_dt = datetime.strptime(end, '%m-%d-%Y-%H:%M')

    avail_scans = get_files(base_path)

    for day in _datetime_range(start_dt, end_dt):

        for scan in avail_scans:
            date = scan.scan_date
            time = scan.scan_time
            if _is_within_range(start_dt, end_dt, datetime.strptime(date + '-' + time, '%m-%d-%Y-%H:%M')):
                if (scan.filename not in added):
                    scans.append(scan)
                    added.append(scan.filename)

    scans.sort(key=lambda x: x.filename)

    return scans



def _datetime_range(start, end):

    diff = (end + timedelta(minutes = 1)) - start

    for x in range(int(diff.total_seconds() / 60)):
        yield start + timedelta(minutes = x)



def _is_within_range(start, end, value):

    if value >= start and value <= end:
        return True
    else:
        return False
