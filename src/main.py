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
import plotting_funcs



def main():
    local_abi_path = '/media/mnichol3/pmeyers1/MattNicholson/goes'
    local_wtlma_path = '/media/mnichol3/pmeyers1/MattNicholson/wtlma'
    local_glm_path = '/media/mnichol3/pmeyers1/MattNicholson/glm'

    abs_path_glm = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523/IXTR99_KNES_232121_40312.2019052321'
    abs_path_wtlma = '/media/mnichol3/pmeyers1/MattNicholson/wtlma/2019/05/23/LYLOUT_190523_212000_0600.dat'

    point1 = (37.195, -102.185)
    point2 = (34.565, -99.865)

    wtlma_data = wtlma.parse_file(abs_path_wtlma, sub_t='21:21')
    glm_data = glm_utils.read_file(abs_path_glm, meta=True)
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)

    plotting_funcs.plot_mercator_dual(glm_data, (point1, point2), wtlma_data)

    """
    conn = goesawsinterface.GoesAWSInterface()
    imgs = conn.get_avail_images_in_range('goes16', 'ABI-L2-CMIPM', '5-23-2019-20:00', '5-23-2019-21:00', 'M1', '13')
    for x in imgs:
        print(x)

    avail_glm_imgs = localglminterface.get_files_in_range('/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523', '5-23-2019-20:00', '5-23-2019-21:00')
    for x in avail_glm_imgs:
        print(x)
    """





if (__name__ == '__main__'):
    main()
