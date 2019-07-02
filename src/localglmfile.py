"""
Author: Matt Nicholson

A simple class for local GOES-16 Geostationary Lightning Mapper (GLM) netCDF files
"""
import os
import re


class LocalGLMFile(object):

    def __init__(self, abs_path):
        super(LocalGLMFile, self).__init__()
        self._scan_date_re = re.compile(r'\.(\d{8})')
        self._scan_time_re = re.compile(r'_\d{2}(\d{4})_')
        self.abs_path = abs_path
        self.filename = None
        self.scan_date = None
        self.scan_time = None
        self.data = None
        if (abs_path is not None):
            self._parse_fname(abs_path)
            self._parse_scan_date()
            self._parse_scan_time()



    def set_data(self, new_data):
        self.data = new_data



    def _parse_fname(self, abs_path):
        _, self.filename = os.path.split(abs_path)



    def _parse_scan_date(self):
        if (self.filename is not None):
            match = self._scan_date_re.search(self.filename)

            if (match):
                date = match.group(1)

                year = date[0:4]
                month = date[4:6].zfill(2)
                day = date[6:8].zfill(2)

                self.scan_date = month + '-' + day + '-' + year
        else:
            raise ValueError('Filepath cannot be None')



    def _parse_scan_time(self):
        if (self.filename is not None):
            match = self._scan_time_re.search(self.filename)

            if (match):
                time = match.group(1)

                hour = time[0:2].zfill(2)
                mins = time[2:4].zfill(2)

                self.scan_time = hour + ":" + mins
        else:
            raise ValueError('Filepath cannot be None')



    def __repr__(self):
        return '<LocalGLMFile object - {}-{}>'.format(self.scan_date, self.scan_time)
