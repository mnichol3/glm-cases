class MRMSComposite(object):
    """
    Class for the MRMSGrib object
    """

    def __init__(self, validity_date, validity_time, major_axis, minor_axis, data_path, fname, shape, grid_lons=None, grid_lats=None):
        """
        Initializes a new MRMSComposite object

        Parameters
        ----------
        validity_date : int or str
        validity_time : int or str
        major_axis : int or str
        minor_axis : int or str
        data_path : str
        fname : str
        shape : tuple
        grid_lons : list
        grid_lats : list

        Attributes
        ----------
        validity_date : int or str
            Validity date of the MRMS grib file
        validity_time : int or str
            Validity time of the MRMS grib file
        major_axis : int or str
            Major axis of projection
        minor_axis : int or str
            Minor axis of projection
        data_path : str
            Path of the memory-mapped array file containing the MRMS data array
        fname : str
            Name of the MRMS grib file as it exists in the parent directory
        shape : tuple
            Shape of the MRMS data array
        grid_lons : list of float
            Grid longitude coordinates
        grid_lats : list of float
            Grid latitude coordinates

        """
        memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'

        super(MRMSComposite, self).__init__()
        self.validity_date = validity_date
        self.validity_time = validity_time
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.data_path = data_path
        self.fname = fname
        self.shape = shape
        self.grid_lons = grid_lons
        self.grid_lats = grid_lats



    def get_data_path(self):
        return self.data_path



    def __repr__(self):
        return '<MRMSComposite object - {}z>'.format(str(self.validity_date) + '-' + str(self.validity_time))
