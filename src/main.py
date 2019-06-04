from aws_dl import abi_dl
import boto3
from botocore.handlers import disable_signing
import goesawsinterface
import sys

"""
date_time = '201905232107'
sector = 'meso1'

abi_dl(date_time, sector)
"""

"""

s3conn = boto3.resource('s3')
s3conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket = s3conn.Bucket('noaa-goes16')
resp = bucket.meta.client.list_objects(Bucket='noaa-goes16', Delimiter='/')
print(resp)
print('\n')



for x in resp.get('CommonPrefixes'):
    print(list(x.values())[0][:-1])

for y in resp.get('Contents'):
    print(y)

prefix = 'ABI-L2-CMIPM/'
print(prefix)

resp = bucket.meta.client.list_objects(Bucket='noaa-goes16', Prefix=prefix, Delimiter='/')
print(resp)

"""

conn = goesawsinterface.GoesAWSInterface()

products = conn.get_avail_products('goes16')
print(products)


years = conn.get_avail_years('goes16', 'ABI-L2-CMIPM')
print(years)


days = conn.get_avail_days('goes16', 'ABI-L2-CMIPM', 2018)
print(days)
