import os
import re
import sys
from datetime import timedelta, datetime

import boto3
import errno
import pytz
import six
from botocore.handlers import disable_signing
import concurrent.futures

from awsgoesfile import AwsGoesFile


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



    def get_avail_images(self, satellite, product, date, sector, channel):
        images = []

        if (not isinstance(date, datetime)):
            date = datetime.strptime(date, '%m-%d-%Y-%H')

        year = date.year
        hour = date.hour
        jul_day = date.timetuple().tm_yday

        prefix = self.build_prefix(product, year, jul_day, hour)
        resp = self.get_sat_bucket(satellite, prefix)

        for each in list(resp['Contents']):

            match = self._scan_re.search(each['Key'])
            if (match is not None):
                if (sector in match.group(1) and channel in match.group(1)):
                    time = match.group(2)
                    dt = datetime.strptime(str(year) + ' ' + str(jul_day) + ' ' + time, '%Y %j %H%M')
                    dt = dt.strftime('%m-%d-%Y-%H:%M')
                    images.append(AwsGoesFile(each['Key'], match.group(1) + ' ' + dt, dt))

        return images



    """
    start : str
        format: 'MM-DD-YYYY-HHMM'
    end : str
        format: 'MM-DD-YYYY-HHMM'
    """
    def get_avail_images_in_range(self, satellite, product, start, end, sector, channel):
        images = {}

        start_dt = datetime.strptime(start, '%m-%d-%Y-%H:%M')
        end_dt = datetime.strptime(end, '%m-%d-%Y-%H:%M')

        for day in self.datetime_range(start_dt, end_dt):

            avail_imgs = self.get_avail_images(satellite, product, day, sector, channel)

            for img in avail_imgs:
                if (self.build_channel_format(channel) in img.shortfname and sector in img.shortfname):
                    if self._is_within_range(start_dt, end_dt, datetime.strptime(img.scan_time, '%m-%d-%Y-%H:%M')):
                        images[img.shortfname] = img

        return images



    def download(self, satellite, awsgoesfiles, basepath, keep_aws_folders=False, threads=6):

        if type(awsgoesfiles) == AwsGoesFile:
            awsgoesfiles = [awsgoesfiles]

        localfiles = []
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_download = {executor.submit(self._download,goesfile,basepath,keep_aws_folders,satellite): goesfile for goesfile in awsgoesfiles}

            for future in concurrent.futures.as_completed(future_download):
                try:
                    result = future.result()
                    localfiles.append(result)
                    six.print_("Downloaded {}".format(result.filename))
                except GoesAwsDownloadError:
                    error = future.exception()
                    errors.append(error.awsgoesfile)

        # Sort returned list of NexradLocalFile objects by the scan_time
        localfiles.sort(key=lambda x:x.scan_time)
        downloadresults = DownloadResults(localfiles,errors)
        six.print_('{} out of {} files downloaded...{} errors'.format(downloadresults.success_count,
                                                                      downloadresults.total,
                                                                      downloadresults.failed_count))
        return downloadresults



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



    def build_channel_format(self, channel):
        if not isinstance(channel, str):
            channel = str(channel)

        return 'C' + channel.zfill(2)



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



    def datetime_range(self, start, end):
        diff = (end + timedelta(minutes = 1)) - start

        for x in range(int(diff.total_seconds() / 60)):
            yield start + timedelta(minutes = x)



    def _is_within_range(self, start, end, value):
        if value >= start and value <= end:
            return True
        else:
            return False



    def parse_partial_fname(self, satellite, product, sector, channel, date):

        if (date.year > 2018):
            mode = 'M6'
        else:
            mode = 'M3'

        year = str(date.year)
        day = str(date.timetuple().tm_yday)
        hour = str(date.hour)
        minute = str(date.minute)

        fname = 'ABI-L2-' + product + '/' + year + '/' + day
        fname += '/OR_ABI-L2-' + product + sector + '-' + mode
        fname += self.build_channel_format(channel) + '_G' + satellite[-2:] + '_'
        fname += 's' + year + day + hour + minute

        return fname



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



    def _download(self, awsgoesfile, basepath, keep_aws_folders, satellite):
        dirpath, filepath = awsgoesfile.create_filepath(basepath, keep_aws_folders)

        try:
            os.makedirs(dirpath)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
                pass
            else:
                raise

        try:
            s3 = boto3.client('s3')
            s3.meta.events.register('choose-signer.s3.*', disable_signing)
            if (satellite = 'goes16'):
                bucket = 'noaa-goes16'
            elif (satellite = 'goes17'):
                bucket = 'noaa-goes17'
            else:
                print('Error: Invalid satellite')
                sys.exit(0)

            s3.download_file(bucket, awsgoesfile.key, filepath)
            return LocalGoesFile(awsgoesfile, filepath)
        except:
            message = 'Download failed for {}'.format(awsgoesfile.shortfname)
            raise GoesAwsDownloadError(message, awsgoesfile)



class GoesAwsDownloadError(Exception):
    def __init__(self, message, awsgoesfile):
        super(GoesAwsDownloadError, self).__init__(message)
        self.awsgoesfile = awsgoesfile
