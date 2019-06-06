import os
import re

from datetime import datetime


class AwsGoesFile(object):

    def __init__(self, key, shortfname, scan_time):
        super(AwsGoesFile, self).__init__()
        #self._scan_time_re = re.compile(r'(....)(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}).*')
        self.key = key
        self.shortfname = shortfname
        self.scan_time = scan_time



    def __repr__(self):
        return '<AwsGoesFile object - {}>'.format(self.shortfname)
