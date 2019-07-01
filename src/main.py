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
import plotting_utils



def main():
    local_abi_path = '/media/mnichol3/pmeyers1/MattNicholson/goes'
    local_wtlma_path = '/media/mnichol3/pmeyers1/MattNicholson/wtlma'
    local_glm_path = '/media/mnichol3/pmeyers1/MattNicholson/glm'
    local_mrms_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'

    abs_path_glm = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523/IXTR99_KNES_232121_40312.2019052321'
    abs_path_wtlma = '/media/mnichol3/pmeyers1/MattNicholson/wtlma/2019/05/23/LYLOUT_190523_212000_0600.dat'


    # 2120
    #point1 = (37.195, -102.185)
    #point2 = (34.565, -99.865)

    # 21:19
    point1 = (35.565, -101.365)
    point2 = (36.045, -101.115)


    wtlma_data = wtlma.parse_file(abs_path_wtlma, sub_t='21:19')
    wtlma_data._set_data(plotting_utils.filter_by_dist(wtlma_data.data, 3000, point1, point2, 100))

    #plotting_funcs.run_mrms_xsect(local_mrms_path, '2119', point1, point2)

    cross_data, lats, lons = plotting_utils.process_slice(local_mrms_path, '2119', point1, point2)
    plotting_funcs.plot_mrms_cross_section2(data=cross_data, lons=lons, lats=lats, wtlma_df=wtlma_data.data)


    #plotting_funcs.plot_mrms_cross_section2(data=None, abs_path=None, lons=None, lats=None, wtlma_df=None)
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)
    #glm_data = glm_utils.read_file(abs_path_glm, meta=True, window=True)

    #plotting_funcs.plot_mercator_dual(glm_data, (point1, point2), wtlma_data)
    #plotting_funcs.plot_mercator_dual_2(glm_data, (point1, point2), wtlma_data)


    """
    base_path = '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905'
    plotting_funcs.run_mrms_xsect(base_path, '2124', point1, point2)
    """

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
