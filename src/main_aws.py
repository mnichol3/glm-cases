
import goesawsinterface
import sys
import six

basepath = '/media/mnichol3/pmeyers1/MattNicholson/goes/'

conn = goesawsinterface.GoesAWSInterface()

#prods = conn.get_avail_products('goes16')
#print(conn.get_avail_years('goes16', 'ABI-L1b-RadM'))
#imgs = conn.get_avail_images('goes16', 'ABI-L1b-RadM', '5-23-2019-21', 'M2', '13')

"""
imgs = conn.get_avail_images_in_range('goes16', 'ABI-L2-CMIPM', '5-23-2019-20:00', '5-23-2019-21:00', 'M1', '13')
for img in imgs:
    print(img)
"""

#imgs = conn.get_avail_images_in_range('goes16', 'ABI-L2-CMIPM', '5-23-2019-20:00', '5-23-2019-21:00', 'M1', '13')
#localfiles = conn.download('goes16', imgs, basepath)
#six.print_(localfiles.success)
#six.print_(localfiles.success[0].filepath)
