import sys
import six
import pandas as pd
from sys import exit, getrefcount
from datetime import datetime
from os.path import isfile
from os import remove

import recipes



def main():
    case_coords = '/home/mnichol3/Coding/glm-cases/resources/05232019-coords.txt'
    wwa_fname = ('/home/mnichol3/Coding/glm-cases/resources/wwa_201905230000_201905240000'
                 '/wwa_201905230000_201905240000.shp')

    func_name = 'wtlma_glm_mercator_dual'
    func_ext = 'hitemp'

    # func_name = 'plot_merc_abi_mrms'
    # func_ext = None

    # func_name = 'mrms_xsect2'
    # func_ext = None
    func_mode = 2

    extent = [35.362, 36.992, -102.443, -100.00]

    sat_meta = {
        'satellite': 'goes16',
        'vis_prod': 'ABI-L1b-Rad',
        'inf_prod': 'ABI-L2-CMIP',
        'sector': 'M2',
        'vis_chan': '2',
        'inf_chan': '13',
        'glm_5min': False
    }

    plot_sets = {
        'show': False,
        'save': True
    }

    paths = _pat_paths()
    paths['wwa'] = wwa_fname

    if (isfile(paths['logpath'])):
        remove(paths['logpath'])

    recipes.driver(paths, case_coords, extent, sat_meta, func_name, plot_sets,
               func_ext=func_ext, func_mode=func_mode, points=None)



def _matt_paths():
    paths = {
       'local_abi_path': '/media/mnichol3/tsb1/data/abi',
       'local_wtlma_path': '/media/mnichol3/tsb1/data/wtlma',
       'local_glm_path': '/media/mnichol3/tsb1/data/glm',
       'local_mrms_path': '/media/mnichol3/tsb1/data/mrms/201905',
       'memmap_path': '/media/mnichol3/tsb1/data/data',
       'img_outpath': '/home/mnichol3/Coding/glm-cases/imgs/05232019/auto-out',
       'logpath': '/home/mnichol3/Coding/glm-cases/misc/runlog.txt'
    }
    return paths



def _pat_paths():
    paths = {
        'local_abi_path': '/media/mnichol3/pmeyers1/MattNicholson/abi',
        'local_wtlma_path': '/media/mnichol3/pmeyers1/MattNicholson/wtlma',
        'local_glm_path': '/media/mnichol3/pmeyers1/MattNicholson/glm',
        'local_mrms_path': '/media/mnichol3/pmeyers1/MattNicholson/mrms/201905',
        'memmap_path': '/media/mnichol3/pmeyers1/MattNicholson/data',
        'img_outpath': '/home/mnichol3/Coding/glm-cases/imgs/05232019/auto-out',
        'logpath': '/home/mnichol3/Coding/glm-cases/misc/runlog.txt'
    }
    return paths


if (__name__ == '__main__'):
    main()
