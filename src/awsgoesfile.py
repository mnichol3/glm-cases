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



    def create_filepath(self, basepath, keep_aws_structure):
        """
        This function creates the file path in preperation for downloading. If keep_aws_structure
        is True then subfolders will be created under the basepath with the same structure as the
        AWS Nexrad Bucket.
        You should not need to call this function as it is done for you on download.
        :param basepath: string - base folder to save files too
        :param keep_aws_structure: boolean - weather or not to use the aws folder structure
         inside the basepath...(year/month/day/radar/)
        :return: tuple - directory path and full filepath
        """
        if keep_aws_structure:
            directorypath = os.path.join(basepath, self.awspath)
            filepath = os.path.join(directorypath, self.filename)
        else:
            directorypath = basepath
            filepath = os.path.join(basepath, self.filename)

        return directorypath,filepath




    def __repr__(self):
        return '<AwsGoesFile object - {}>'.format(self.shortfname)
