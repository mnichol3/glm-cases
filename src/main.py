"""
Author: Matt Nicholson

"""
import sys
import six

import wtlma
import goesawsinterface
import grib
import localglminterface
import glm_utils

def main():
    local_abi_path = '/media/mnichol3/pmeyers1/MattNicholson/goes'
    local_wtlma_path = '/media/mnichol3/pmeyers1/MattNicholson/wtlma'
    local_glm_path = '/media/mnichol3/pmeyers1/MattNicholson/glm'

    conn = goesawsinterface.GoesAWSInterface()
    imgs = conn.get_avail_images_in_range('goes16', 'ABI-L2-CMIPM', '5-23-2019-20:00', '5-23-2019-21:00', 'M1', '13')
    for x in imgs:
        print(x)

    avail_glm_imgs = localglminterface.get_files_in_range('/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523', '5-23-2019-20:00', '5-23-2019-21:00')
    for x in avail_glm_imgs:
        print(x)





if (__name__ == '__main__'):
    main()
