from os.path import isfile
from netCDF4 import Dataset


def trim_header(abs_path):
    if (not isfile(abs_path)):
        raise OSError('File does not exist:', abs_path)

    out_path = abs_path + '.nc'

    if (not isfile(out_path)):

        with open(abs_path, 'rb') as f_in:
            f_in.seek(21)
            data = f_in.read()
            f_in.close()
            f_in = None

        with open(out_path, 'wb') as f_out:
            f_out.write(data)
            f_out.close()
            f_out = None

    return out_path



def print_file_format(abs_path):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    print(fh.file_format)



def print_dimensions(abs_path):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    for dim in fh.dimensions.keys():
        print(dim)



def print_variables(abs_path):
    f_path = trim_header(abs_path)

    fh = Dataset(f_path, 'r')

    for var in fh.variables.keys():
        print(var)



fname = '/media/mnichol3/pmeyers1/MattNicholson/glm/glm20190523/IXTR99_KNES_231752_13933.2019052318'
print_variables(fname)
