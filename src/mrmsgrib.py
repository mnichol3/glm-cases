import re

class MRMSGrib(object):

    def __init__(self, validity_date, validity_time, data, major_axis, minor_axis, abs_path):
        super(MRMSGrib, self).__init__()
        self.validity_date = validity_date
        self.validity_time = validity_time
        self.data = data
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.grid_lons = None
        self.grid_lats = None
        self.path = None
        self.fname = None
        self.scan_angle = None
        if abs_path is not None:
            self.parse_path(abs_path)
        self.parse_scan_angle(self.fname)



    def parse_path(self, abs_path):
        self.path, self.fname = abs_path.rsplit('/', 1)



    def parse_scan_angle(self, fname):
        scan_re = re.compile(r'_(\d{2}.\d{2})_')
        match = scan_re.search(fname)

        if (match is not None):
            self.scan_angle = match.group(1)



    def set_data(self, new_data):
        self.data = new_data



    def set_grid_lons(self, lons):
        self.grid_lons = lons



    def set_grid_lats(self, lats):
        self.grid_lats = lats



    def metadata(self):
        print('------------------------------------')
        print('validity date:', self.validity_date)
        print('validity_time:', self.validity_time)
        print('scan angle:', self.scan_angle)
        print('major_axis:', self.major_axis)
        print('minor_axis:', self.minor_axis)
        print('file path:', self.path)
        print('file name:', self.fname)
        print('------------------------------------')
        print('\n')



    def __repr__(self):
        return '<MRMSGrib object - {}z>'.format(str(self.scan_angle) + '-' + str(self.validity_date) + '-' + str(self.validity_time))
