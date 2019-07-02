"""
Author: Matt Nicholson

A simple class for local West Texas LMA (WTLMA) files
"""
from os.path import join, isfile, split
from os import listdir, walk
import sys
import re

class LocalWtlmaFile(object):

    BASE_PATH = '/media/mnichol3/pmeyers1/MattNicholson/wtlma'

    def __init__(self, abs_path, start_time, coord_center, max_diameter, active_stations):
        super(LocalWtlmaFile, self).__init__()
        self.filename = None
        self.abs_path = abs_path
        self.start_time = start_time
        self.coord_center = coord_center
        self.max_diameter = max_diameter
        self.active_stations = active_stations
        self.data = None
        if (abs_path is not None):
            self.filename = self._parse_fname(abs_path)



    def _set_data(self, data):
        self.data = data



    def _parse_fname(self, abs_path):
        _, self.filename = split(abs_path)



    def _start_time_pp(self):
        date, time = self.start_time.split('-')
        time = time[:2] + ':' + time[2:]
        year = date[:4]
        month = date[4:6]
        day = date[-2:]
        pp = month + '-' + day + '-' + year + ' ' + time
        return pp



    def _data_start_pp(self):
        date, _ = self.start_time.split('-')
        year = date[:4]
        month = date[4:6]
        day = date[-2:]
        time = self.data['time'].iloc[0]
        pp = month + '-' + day + '-' + year + ' ' + time
        return pp



    def _data_end_pp(self):
        date, _ = self.start_time.split('-')
        year = date[:4]
        month = date[4:6]
        day = date[-2:]
        time = self.data['time'].iloc[-1]
        pp = month + '-' + day + '-' + year + ' ' + time
        return pp


    def __repr__(self):
        return '<LocalWtlmaFile object - {}>'.format(self.start_time)
