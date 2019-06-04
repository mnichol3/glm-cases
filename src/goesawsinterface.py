import os
import re
import sys
from datetime import timedelta

import boto3
import errno
import pytz
import six
from botocore.handlers import disable_signing


"""
import goesawsinterface

conn = goesawsinterface.GoesAWSInterface()


years = conn.get_avail_products('goes16')
print(years)


years = conn.get_avail_years('goes16', 'ABI-L1b-RadC')
print(years)
"""

class GoesAWSInterface(object):
    """
    Instantiate an instance of this class to get a connection to the Nexrad AWS bucket. \
    This class provides methods to query for various metadata of the AWS bucket as well \
    as download files.
    >>> import nexradaws
    >>> conn = nexradaws.NexradAwsInterface()
    """
    def __init__(self):
        super(GoesAWSInterface, self).__init__()
        self._year_re = re.compile(r'/(\d{4})/')
        self._day_re = re.compile(r'/\d{4}/(\d{3})/')
        self._radar_re = re.compile(r'^\d{4}/\d{2}/\d{2}/(....)/')
        self._scan_re = re.compile(r'^\d{4}/\d{2}/\d{2}/..../(?:(?=(.*.gz))|(?=(.*V0*.gz))|(?=(.*V0*)))')
        self._s3conn = boto3.resource('s3')
        self._s3conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        self._bucket_16 = self._s3conn.Bucket('noaa-goes16')
        self._bucket_17 = self._s3conn.Bucket('noaa-goes17')



    def get_avail_products(self, satellite):
        prods = []

        resp = self.get_sat_bucket(satellite, '')

        for x in resp.get('CommonPrefixes'):
            #print(list(x.values())[0][:-1])
            prods.append(list(x.values())[0][:-1])

        return prods



    def get_avail_years(self, satellite, product):
        years = []
        prefix = self.build_prefix(product)

        resp = self.get_sat_bucket(satellite, prefix)

        for each in resp.get('CommonPrefixes'):
            match = self._year_re.search(each['Prefix'])
            if (match is not None):
                years.append(match.group(1))

        return years



    def get_avail_days(self, satellite, product, year):
        days = []
        prefix = self.build_prefix(product, year)

        resp = self.get_sat_bucket(satellite, prefix)
        for each in resp.get('CommonPrefixes'):
            match = self._day_re.search(each['Prefix'])
            if (match is not None):
                days.append(match.group(1))

        return days



    def build_prefix(self, product=None, year=None, julian_day=None, hour=None):
        prefix = ''

        if product is not None:
            prefix += product
            prefix += '/'
        if year is not None:
            prefix += self.build_year_format(year)
        if julian_day is not None:
            prefix += self.build_day_format(julian_day)
        if hour is not None:
            prefix += self.build_hour_format(hour)

        return prefix



    def build_year_format(self, year):
        if (isinstance(year, int)):
            return '{:04}/'.format(year)
        elif (isinstance(year, str)):
            return '{}/'.format(year)
        else:
            raise TypeError('Year must be of type int or str')



    def build_day_format(self, jd):
        if isinstance(jd, int):
            return '{:03}/'.format(jd)
        elif isinstance(jd, str):
            return '{}/'.format(m_or_d)
        else:
            raise TypeError('Month must be int or str type')



    def get_sat_bucket(self, satellite, prefix):
        resp = None

        if (satellite == 'goes16'):
            resp = self._bucket_16.meta.client.list_objects(Bucket='noaa-goes16', Prefix=prefix, Delimiter='/')
        elif (satellite == 'goes17'):
            resp = self._bucket_17.meta.client.list_objects(Bucket='noaa-goes17', Prefix=prefix, Delimiter='/')
        else:
            print('Error: Invalid satellite argument')
            sys.exit(0)

        return resp
