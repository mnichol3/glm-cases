from aws_dl import abi_dl
import boto3
from botocore.handlers import disable_signing
import goesawsinterface
import sys



"""
s3conn = boto3.resource('s3')
s3conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket = s3conn.Bucket('noaa-goes16')
resp = bucket.meta.client.list_objects(Bucket='noaa-goes16', Delimiter='/')

prefix = 'ABI-L2-CMIPM/2018/362/12/'

resp = bucket.meta.client.list_objects(Bucket='noaa-goes16', Prefix=prefix, Delimiter='/')
#print(list(resp))
#print(list(resp['Contents']))
for x in list(resp['Contents']):
    print(x['Key'])


"""



conn = goesawsinterface.GoesAWSInterface()

#days = conn.get_avail_days('goes16', 'ABI-L2-CMIPM', 2018)
#print(days)
#print(conn.decode_julian_day('2018', days))
print(conn.get_avail_images('goes16', 'ABI-L2-CMIPM', '6-6-2019-12', 'M1', '13'))
#print(conn.get_avail_images_in_range('goes16', 'ABI-L2-CMIPM', '5-6-2019-15:05', '5-6-2019-15:25', 'M2', '1'))
