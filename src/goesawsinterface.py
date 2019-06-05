import os
import re
import sys
from datetime import timedelta, datetime

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
        self._hour_re = re.compile(r'/\d{4}/\d{3}/(\d{2})/')
        self._scan_re = re.compile(r'(\w{5}\d?-M\dC\d{2})_G\d{2}_s\d{7}(\d{4})\d{3}')
        self._s3conn = boto3.resource('s3')
        self._s3conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        self._bucket_16 = self._s3conn.Bucket('noaa-goes16')
        self._bucket_17 = self._s3conn.Bucket('noaa-goes17')



    def get_avail_products(self, satellite):
        prods = []

        resp = self.get_sat_bucket(satellite, '')

        for x in resp.get('CommonPrefixes'):
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



    def get_avail_months(self, satellite, product, year):
        days = self.get_avail_days(satellite, product, year)
        months = self.decode_julian_day(year, days, 'm')

        return months



    def get_avail_days(self, satellite, product, year):
        days = []
        prefix = self.build_prefix(product, year)

        resp = self.get_sat_bucket(satellite, prefix)
        for each in resp.get('CommonPrefixes'):
            match = self._day_re.search(each['Prefix'])
            if (match is not None):
                days.append(match.group(1))

        return days



    def get_avail_hours(self, satellite, product, date):
        hours = []

        year = date[-4:]
        jul_day = datetime.strptime(date, '%m-%d-%Y').timetuple().tm_yday

        prefix = self.build_prefix(product, year, jul_day)
        resp = self.get_sat_bucket(satellite, prefix)

        for each in resp.get('CommonPrefixes'):
            match = self._hour_re.search(each['Prefix'])
            if (match is not None):
                hours.append(match.group(1))

        return hours



    def get_avail_images(self, satellite, product, date, hour):
        images = []

        year = date[-4:]
        jul_day = datetime.strptime(date, '%m-%d-%Y').timetuple().tm_yday

        prefix = self.build_prefix(product, year, jul_day, hour)
        resp = self.get_sat_bucket(satellite, prefix)

        for each in list(resp['Contents']):
            match = self._scan_re.search(each['Key'])
            if (match is not None):
                images.append(match.group(1) + '-' + match.group(2))

        return images



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



    def build_hour_format(self, hour):
        if isinstance(hour, int):
            return '{:02}/'.format(hour)
        elif isinstance(hour, str):
            return '{}/'.format(hour)
        else:
            raise TypeError('Hour must be int or str type')



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



    def decode_julian_day(self, year, days, key):
        dates = {}

        if not isinstance(year, str):
            year = str(year)

        for day in days:
            curr = datetime.strptime(year[2:] + day, '%y%j').date()

            if (curr.month in list(dates)):
                dates[curr.month].append(curr.day)
            else:
                dates[curr.month] = [curr.day]

        if (key == 'm'):
            return list(dates)
        else:
            return dates
