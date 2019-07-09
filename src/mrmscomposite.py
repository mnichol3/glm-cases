class MRMSComposite(object):
    """
    Class for the MRMSGrib object
    """

    def __init__(self, validity_date, validity_time, major_axis, minor_axis, data_path, fname, shape, grid_lons=None, grid_lats=None):
        """
        Initializes a new MRMSGrib object

        Parameters
        ----------
        validity_date : int or str
        validity_time : int or str
        data : numpy 2d array
        major_axis : int or str
        minor_axis : int or str
        abs_path : str

        Attributes
        ----------
        validity_date : int or str
            Validity date of the MRMS grib file
        validity_time : int or str
            Validity time of the MRMS grib file
        data : numpy 2d array
            MRMS reflectivity data
        major_axis : int or str
            Major axis of projection
        minor_axis : int or str
            Minor axis of projection
        grid_lons : list of float
            Grid longitude coordinates
        grid_lats : list of float
            Grid latitude coordinates
        path : str
            Path of the MRMS grib file
        fname : str
            Name of the MRMS grib file as it exists in the parent directory
        scan_angle : str or float
            Scan angle of the MRMS reflectivity data

        """
        memmap_path = '/media/mnichol3/pmeyers1/MattNicholson/data'

        super(MRMSGrib, self).__init__()
        self.validity_date = validity_date
        self.validity_time = validity_time
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.data_path = data_path
        self.fname = fname
        self.shape = shape
        self.grid_lons = grid_lons
        self.grid_lats = grid_lats
