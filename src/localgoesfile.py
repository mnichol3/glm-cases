import os
from netCDF4 import Dataset


class LocalGoesFile(object):

    def __init__(self, awsgoesfile, localfilepath):
        super(LocalGoesFile, self).__init__()
        self.key = awsgoesfile.key
        self.shortfname = awsgoesfile.shortfname
        self.filename = awsgoesfile.filename
        self.scan_time = awsgoesfile.scan_time
        self.filepath = localfilepath



    def read(self):
        """
        Opens & reads a GOES-16 ABI data file, returning a dictionary of data

        Parameters:
        ------------
        None


        Returns:
        --------
        data_dict : dictionary of str
            Dictionar of ABI image data & metadata from the netCDF file

        Notes
        -----
        !!! UNTESTED !!! 
        """
        print('Reading', __repr__(self))

        data_dict = {}

        fh = Dataset(self.filepath, mode='r')

        data_dict['band_id'] = fh.variables['band_id'][0]

        data_dict['band_wavelength'] = "%.2f" % fh.variables['band_wavelength'][0]
        data_dict['semimajor_ax'] = fh.variables['goes_imager_projection'].semi_major_axis
        data_dict['semiminor_ax'] = fh.variables['goes_imager_projection'].semi_minor_axis
        data_dict['inverse_flattening'] = fh.variables['goes_imager_projection'].inverse_flattening
        data_dict['latitude_of_projection_origin'] = fh.variables['goes_imager_projection'].latitude_of_projection_origin
        data_dict['longitude_of_projection_origin'] = fh.variables['goes_imager_projection'].longitude_of_projection_origin
        data_dict['data_units'] = fh.variables['CMI'].units

        # Seconds since 2000-01-01 12:00:00
        add_seconds = fh.variables['t'][0]

        # Datetime of scan
        scan_date = datetime(2000, 1, 1, 12) + timedelta(seconds=float(add_seconds))

        # Satellite height
        sat_height = fh.variables['goes_imager_projection'].perspective_point_height

        # Satellite longitude & latitude
        sat_lon = fh.variables['goes_imager_projection'].longitude_of_projection_origin
        sat_lat = fh.variables['goes_imager_projection'].latitude_of_projection_origin

        # Satellite lat/lon extend
        lat_lon_extent = {}
        lat_lon_extent['n'] = fh.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude
        lat_lon_extent['s'] = fh.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
        lat_lon_extent['e'] = fh.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
        lat_lon_extent['w'] = fh.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude

        # Geospatial lat/lon center
        data_dict['lat_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lat_center
        data_dict['lon_center'] = fh.variables['geospatial_lat_lon_extent'].geospatial_lon_center

        # Satellite sweep
        sat_sweep = fh.variables['goes_imager_projection'].sweep_angle_axis

        data = fh.variables['CMI'][:].data

        Xs = fh.variables['x'][:]
        Ys = fh.variables['y'][:]

        fh.close()
        fh = None

        data_dict['scan_date'] = scan_date
        data_dict['sat_height'] = sat_height
        data_dict['sat_lon'] = sat_lon
        data_dict['sat_lat'] = sat_lat
        data_dict['lat_lon_extent'] = lat_lon_extent
        data_dict['sat_sweep'] = sat_sweep
        data_dict['x'] = Xs
        data_dict['y'] = Ys
        data_dict['data'] = data

        return data_dict



    def __repr__(self):
        return '<LocalGoesFile object - {}>'.format(self.filepath)
