import os
import re

from datetime import datetime


class AwsGoesFile(object):

    def __init__(self, key, shortfname, scan_time):
        super(AwsGoesFile, self).__init__()
        self.key = key
        self.shortfname = shortfname
        self.scan_time = scan_time
        self.awspath = None
        self.filename = None
        if self.key is not None:
            self._parse_key()



    def _parse_key(self):
        self.awspath, self.filename = os.path.split(self.key)



    def __repr__(self):
        return '<AwsGoesFile object - {}>'.format(self.shortfname)
