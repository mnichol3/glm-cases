

class GribObj(object):

    def __init__(self, validity_date, validity_time, data, major_axis, minor_axis):
        super(GribObj, self).__init__()
        self.validity_date = validity_date
        self.validity_time = validity_time
        self.data = data
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.grid_lons = None
        self.grid_lats = None



    def set_data(self, new_data):
        self.data = new_data



    def set_grid_lons(self, lons):
        self.grid_lons = lons



    def set_grid_lats(self, lats):
        self.grid_lats = lats


        
    def __repr__(self):
        return '<Grib object - {}>'.format(self.validity_date + '-' + self.validity_time)
