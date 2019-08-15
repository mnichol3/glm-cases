"""
Author: Matt Nicholson

A class for local Multi-Radar/Multi-Sensor System (MRMS) GRIB files
"""
import re
from os.path import join

class MRMSGrib(object):
    """
    Class for the MRMSGrib object
    """

    def __init__(self, validity_date, validity_time, major_axis, minor_axis, path, fname, shape, grid_lons=None, grid_lats=None):
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
        self.data_path = join(memmap_path, fname.replace('grib2', 'txt'))
        self.path = path
        self.fname = fname
        self.shape = shape
        self.grid_lons = grid_lons
        self.grid_lats = grid_lats
        self.scan_angle = None
        self.parse_scan_angle(join(self.path, self.fname))



    def parse_path(self, abs_path):
        """
        Parses the path and the filename of the MRMSGrib file object

        Parameters
        ----------
        abs_path : str

        Returns
        -------
        None, however sets the MRMSGrib object's path & fname attributes

        """
        self.path, self.fname = abs_path.rsplit('/', 1)



    def parse_scan_angle(self, fname):
        """
        Parses the MRMSGrib object file's scan angle

        Parameters
        ----------
        fname : str
            Name of the MRMSGrib file

        Returns
        -------
        None, sets the MRMSGrib object's scan_angle attribute

        """
        scan_re = re.compile(r'_(\d{2}.\d{2})_')
        match = scan_re.search(fname)

        if (match is not None):
            self.scan_angle = match.group(1)



    def set_data(self, new_data):
        """
        Deletes the data currently held by the MRMSGrib object's data field and
        replaces it

        Parameters
        ----------
        new_data : numpy 2d array

        Returns
        -------
        None, sets the MRMSGrib object's data attribute

        """
        del self.data
        self.data = None
        self.data = new_data



    def set_grid_lons(self, lons):
        """
        Parameters
        ----------

        Returns
        -------

        """
        self.grid_lons = lons



    def set_grid_lats(self, lats):
        """
        Parameters
        ----------

        Returns
        -------

        """
        self.grid_lats = lats



    def get_data_path(self):
        return self.data_path



    def metadata(self):
        """
        Prints MRMSGrib object metadata

        """
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
        return '<MRMSGrib object - {}-{}-{}z>'.format(self.scan_angle, self.validity_date,
                                                      self.validity_time)
