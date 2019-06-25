"""
Author: Matt Nicholson

"""
import wtlma
import goesawsinterface
import grib
import localglminterface
import glm_utils

def main():
    avail_glm_imgs = localglminterface.get_files_in_range('/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523', '5-23-2019-20:00', '5-23-2019-21:00')

if (__name__ == '__main__'):
    main)()
